# ---------------------------------------------
# her_sac_curriculum.py
# ---------------------------------------------
# This script demonstrates curriculum-based training of a robotic agent
# in CausalWorld using Hindsight Experience Replay (HER) and Soft Actor-Critic (SAC).
# It integrates Weights & Biases (wandb) for experiment tracking and logging.
#
# Key Features:
#   - Curriculum learning with progressive environment interventions
#   - Custom callbacks for detailed logging and checkpointing
#   - Enhanced model architecture and training loop
#   - Evaluation using CausalWorld's reaching benchmark
#
# Usage:
#   python her_sac_curriculum.py --log_relative_path ./logs --total_time_steps_per_update 200000
#   python her_sac_curriculum.py --log_relative_path ./logs --evaluate_only
# ---------------------------------------------

## python her_sac_curriculum.py --log_relative_path ./enhanced_reaching_logs --total_time_steps_per_update 200000
## python her_sac_curriculum.py --log_relative_path ./enhanced_reaching_logs --evaluate_only

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from causal_world.task_generators.task import generate_task
from causal_world.envs.causalworld import CausalWorld
from causal_world.wrappers.env_wrappers import HERGoalEnvWrapper
from causal_world.wrappers.curriculum_wrappers import CurriculumWrapper
from causal_world.intervention_actors import (
    GoalInterventionActorPolicy, 
    RandomInterventionActorPolicy, 
    VisualInterventionActorPolicy,
    RigidPoseInterventionActorPolicy, 
    PhysicalPropertiesInterventionActorPolicy, 
    JointsInterventionActorPolicy
)
from causal_world.evaluation.evaluation import EvaluationPipeline
from causal_world.benchmark.benchmarks import REACHING_BENCHMARK

from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MultiInputPolicy
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import set_random_seed

import numpy as np
import argparse
import json
import torch
import torch.nn as nn
import wandb
from wandb.integration.sb3 import WandbCallback
import time
from datetime import datetime

# --- Enhanced Callbacks ---

# (Each callback below is responsible for a specific aspect of logging or curriculum management)

class WandbMetricsCallback(BaseCallback):
    """Custom callback to log detailed metrics to W&B during training and evaluation."""
    def __init__(self, eval_env, eval_freq=10000, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        self.curriculum_stage = 0
        
    def _on_step(self) -> bool:
        # Log training metrics every step
        if 'episode' in self.locals:
            episode_info = self.locals.get('episode', [])
            if len(episode_info) > 0:
                for info in episode_info:
                    if 'r' in info:
                        self.episode_rewards.append(info['r'])
                        wandb.log({
                            'custom/curriculum_stage': self.curriculum_stage,
                            'custom/success_rate': success_rate,
                        })
        
        # Log model metrics
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            for key, value in self.model.logger.name_to_value.items():
                if 'train/' not in key:
                    wandb.log({f'model/{key}': value, 'train/timesteps': self.num_timesteps})
        
        # Comprehensive evaluation every eval_freq steps
        if self.n_calls % self.eval_freq == 0:
            self._run_evaluation()
            
        return True
    
    def _run_evaluation(self):
        """Run comprehensive evaluation and log to W&B"""
        episode_rewards = []
        episode_lengths = []
        success_count = 0
        goal_distances = []
        
        obs = self.eval_env.reset()
        
        for episode in range(20):  # 20 evaluation episodes
            done = False
            total_reward = 0
            episode_length = 0
            episode_success = False
            min_goal_distance = float('inf')
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
                total_reward += reward
                episode_length += 1
                
                # Track goal distance if available
                if 'goal_distance' in info:
                    min_goal_distance = min(min_goal_distance, info['goal_distance'])
                
                # Check for success
                if info.get('is_success', False) or reward > 0.5:
                    episode_success = True
                    
            episode_rewards.append(total_reward)
            episode_lengths.append(episode_length)
            goal_distances.append(min_goal_distance if min_goal_distance != float('inf') else 0)
            
            if episode_success:
                success_count += 1
            obs = self.eval_env.reset()
        
        # Calculate metrics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        success_rate = success_count / 20
        mean_goal_distance = np.mean(goal_distances)
        
        # Log comprehensive evaluation metrics
        wandb.log({
            'eval/mean_reward': mean_reward,
            'eval/std_reward': std_reward,
            'eval/mean_episode_length': mean_length,
            'eval/success_rate': success_rate,
            'eval/mean_goal_distance': mean_goal_distance,
            'eval/max_reward': np.max(episode_rewards),
            'eval/min_reward': np.min(episode_rewards),
            'eval/curriculum_stage': self.curriculum_stage,
            'train/timesteps': self.num_timesteps
        })
        
        if self.verbose > 0:
            print(f"[Eval] Step: {self.num_timesteps}")
            print(f"  Mean Reward: {mean_reward:.3f} ± {std_reward:.3f}")
            print(f"  Success Rate: {success_rate:.3f}")
            print(f"  Mean Goal Distance: {mean_goal_distance:.3f}")


class CurriculumProgressCallback(BaseCallback):
    """Callback to track and log curriculum progress, and advance curriculum stages."""
    def __init__(self, curriculum_wrapper=None, verbose=1):
        super().__init__(verbose)
        self.curriculum_wrapper = curriculum_wrapper
        self.recent_rewards = []
        self.curriculum_stage = 0
        self.stage_start_time = time.time()
        
    def _on_step(self) -> bool:
        # Track recent performance
        if 'episode' in self.locals:
            episode_info = self.locals.get('episode', [])
            if len(episode_info) > 0:
                for info in episode_info:
                    if 'r' in info:
                        self.recent_rewards.append(info['r'])
                        
        if len(self.recent_rewards) > 100:
            self.recent_rewards = self.recent_rewards[-100:]
                
        # Log curriculum metrics every 10k steps
        if self.n_calls % 10000 == 0 and len(self.recent_rewards) > 10:
            mean_reward = np.mean(self.recent_rewards[-50:]) if len(self.recent_rewards) >= 50 else np.mean(self.recent_rewards)
            
            wandb.log({
                'curriculum/stage': self.curriculum_stage,
                'curriculum/recent_mean_reward': mean_reward,
                'curriculum/stage_duration': time.time() - self.stage_start_time,
                'curriculum/total_episodes': len(self.recent_rewards),
                'train/timesteps': self.num_timesteps
            })
            
            # Advance curriculum based on performance
            if mean_reward > 0.3 and len(self.recent_rewards) >= 50:
                self.curriculum_stage += 1
                self.stage_start_time = time.time()
                if self.verbose > 0:
                    print(f"Advancing curriculum to stage {self.curriculum_stage}")
                    
        return True


class EnhancedCheckpointCallback(BaseCallback):
    """Enhanced checkpoint callback with W&B artifact logging for model versioning."""
    def __init__(self, save_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f"rl_model_{self.num_timesteps}")
            self.model.save(model_path)
            
            # Log model as W&B artifact
            artifact = wandb.Artifact(
                name=f"model_checkpoint_{self.num_timesteps}",
                type="model",
                description=f"SAC+HER model checkpoint at {self.num_timesteps} timesteps"
            )
            artifact.add_file(f"{model_path}.zip")
            wandb.log_artifact(artifact)
            
            if self.verbose > 0:
                print(f"Saving model checkpoint to {model_path}")
        return True


# class CustomCheckpointCallback(BaseCallback):
#     def __init__(self, save_freq, save_path, verbose=1):
#         super().__init__(verbose)
#         self.save_freq = save_freq
#         self.save_path = save_path
#         os.makedirs(save_path, exist_ok=True)

#     def _on_step(self) -> bool:
#         if self.n_calls % self.save_freq == 0:
#             model_path = os.path.join(self.save_path, f"rl_model_{self.num_timesteps}")
#             self.model.save(model_path)
#             if self.verbose > 0:
#                 print(f"Saving model checkpoint to {model_path}")
#         return True


# class EnhancedEvalCallback(BaseCallback):
#     def __init__(self, eval_env, eval_freq, log_path=None, verbose=1):
#         super().__init__(verbose)
#         self.eval_env = eval_env
#         self.eval_freq = eval_freq
#         self.log_path = log_path
#         self.best_mean_reward = -np.inf
#         self.success_rates = []

#     def _on_step(self) -> bool:
#         if self.n_calls % self.eval_freq == 0:
#             episode_rewards = []
#             success_count = 0
#             obs = self.eval_env.reset()
            
#             for _ in range(20):  # Increased evaluation episodes
#                 done = False
#                 total_reward = 0
#                 episode_success = False
                
#                 while not done:
#                     action, _ = self.model.predict(obs, deterministic=True)
#                     obs, reward, done, info = self.eval_env.step(action)
#                     total_reward += reward
                    
#                     # Check for task success
#                     if info.get('is_success', False) or reward > 0.5:
#                         episode_success = True
                        
#                 episode_rewards.append(total_reward)
#                 if episode_success:
#                     success_count += 1
#                 obs = self.eval_env.reset()
                
#             mean_reward = np.mean(episode_rewards)
#             success_rate = success_count / 20
#             self.success_rates.append(success_rate)
            
#             if self.verbose > 0:
#                 print(f"[Eval] Step: {self.num_timesteps}, Mean Reward: {mean_reward:.2f}, Success Rate: {success_rate:.2f}")
                
#             if mean_reward > self.best_mean_reward:
#                 self.best_mean_reward = mean_reward
#                 best_path = os.path.join(self.log_path or ".", "best_model")
#                 os.makedirs(best_path, exist_ok=True)
#                 self.model.save(os.path.join(best_path, "best_model"))
#                 if self.verbose > 0:
#                     print(f"New best model saved at {best_path}")
#         return True


# --- Curriculum and Environment Factory ---

def create_curriculum_interventions():
    """Create a progressive curriculum of interventions for the environment.
    Each stage increases task difficulty or diversity."""
    interventions = [
        # Stage 1: Easy goals close to initial position
        GoalInterventionActorPolicy(
            goal_intervention={'goal_position': [0.0, 0.0, 0.05]},
            intervention_type='goal_intervention'
        ),
        
        # Stage 2: Random goals within easy reach
        GoalInterventionActorPolicy(
            goal_intervention={'goal_position': [0.1, 0.1, 0.1]},
            intervention_type='goal_intervention'
        ),
        
        # Stage 3: Vary physical properties slightly
        PhysicalPropertiesInterventionActorPolicy(
            intervention_type='physical_properties',
            intervention_params={'mass': [0.8, 1.2]},
            group='tool'
        ),
        
        # Stage 4: Add visual variations
        VisualInterventionActorPolicy(
            intervention_type='visual',
            intervention_params={'color_variation': True}
        ),
        
        # Stage 5: Joint position variations
        JointsInterventionActorPolicy(
            intervention_type='joints',
            intervention_params={'joint_noise': 0.1},
            group='robot'
        ),
        
        # Stage 6: Full randomization
        RandomInterventionActorPolicy(
            intervention_type='random',
            intervention_strength=0.3
        )
    ]
    
    # Define curriculum schedule: (start_step, end_step, frequency, intervention_idx)
    curriculum_schedule = [
        (0, 500000, 1, 0),           # Easy goals for first 500k steps
        (200000, 1000000, 1, 1),     # Overlap with random goals
        (500000, 1500000, 1, 2),     # Add physical variations
        (800000, 2000000, 1, 3),     # Add visual variations
        (1200000, 2500000, 1, 4),    # Add joint variations
        (1800000, 10000000, 1, 5)    # Full randomization
    ]
    
    return interventions, curriculum_schedule


def make_enhanced_env(task_name, log_dir=None, seed=0, skip_frame=3, max_episode_length=2000, 
                     vis=False, use_curriculum=False):
    """Create a CausalWorld environment with optional curriculum and wrappers.
    - HERGoalEnvWrapper enables HER for sparse-reward tasks.
    - Monitor logs episode stats for analysis.
    """
    # task = generate_task(task_generator_id=task_name)
    task = generate_task(
        task_generator_id=task_name,
        variables_space='space_a_b',  # Use full space for more flexibility
        fractional_reward_weight=1.0,  # Keep volumetric overlap reward
        dense_reward_weights=np.array([100000, 50000, 25000, 10000]),  # 4 values: [main_reward, secondary, tertiary, quaternary]
        default_goal_60=np.array([0.0, 0.0, 0.08]),   # Lower, easier goal for finger 1
        default_goal_120=np.array([0.0, 0.0, 0.08]),  # Lower, easier goal for finger 2  
        default_goal_300=np.array([0.0, 0.0, 0.08]),  # Lower, easier goal for finger 3
        joint_positions=None,  # Use default joint positions
        activate_sparse_reward=False  # Keep dense rewards for better learning
    )
    
    env = CausalWorld(
        task=task,
        skip_frame=skip_frame,
        enable_visualization=vis,
        seed=seed,
        max_episode_length=max_episode_length
    )

    # Add curriculum wrapper if requested
    if use_curriculum:
        interventions, schedule = create_curriculum_interventions()
        env = CurriculumWrapper(
            env,
            intervention_actors=interventions,
            actives=schedule
        )

    # Apply HER wrapper first
    env = HERGoalEnvWrapper(env)
    
    # Apply Monitor wrapper last
    if log_dir is not None:
        env = Monitor(env, log_dir, allow_early_resets=True)
    
    return env

# --- Enhanced Model Architecture ---

class CustomSACPolicy(MultiInputPolicy):
    """Custom SAC policy with a larger neural network for improved learning."""
    def __init__(self, *args, **kwargs):
        # Enhanced network architecture
        kwargs['net_arch'] = {
            'pi': [512, 512, 256],  # Larger policy network
            'qf': [512, 512, 256]   # Larger Q-function network
        }
        kwargs['activation_fn'] = nn.ReLU
        kwargs['normalize_images'] = False
        super().__init__(*args, **kwargs)


# --- Enhanced Training Function ---

def train_enhanced_policy(
    log_relative_path, maximum_episode_length, skip_frame, seed_num,
    sac_config, total_time_steps, validate_every_timesteps, task_name,
    wandb_config=None
):
    """Train a SAC+HER agent with curriculum and log progress to wandb."""
    # Initialize W&B
    if wandb_config:
        wandb.init(
            project=wandb_config['project'],
            name=wandb_config['run_name'],
            config={
                'task_name': task_name,
                'maximum_episode_length': maximum_episode_length,
                'skip_frame': skip_frame,
                'seed_num': seed_num,
                'total_time_steps': total_time_steps,
                'validate_every_timesteps': validate_every_timesteps,
                **sac_config
            },
            tags=['HER', 'SAC', 'curriculum', 'reaching'],
            sync_tensorboard=True,
            monitor_gym=True,
            dir=log_relative_path
        )
    
    os.makedirs(log_relative_path, exist_ok=True)
    log_dir = os.path.join(log_relative_path, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Set random seed
    set_random_seed(seed_num)

    # Create enhanced environments
    env = make_enhanced_env(
        task_name, log_dir, seed=seed_num, skip_frame=skip_frame, 
        max_episode_length=maximum_episode_length, use_curriculum=True
    )
    eval_env = make_enhanced_env(
        task_name, None, seed=seed_num+100, skip_frame=skip_frame, 
        max_episode_length=maximum_episode_length, use_curriculum=False
    )

    # Enhanced replay buffer configuration
    replay_buffer_kwargs = {
        'n_sampled_goal': sac_config.pop('n_sampled_goal', 4),
        'goal_selection_strategy': sac_config.pop('goal_selection_strategy', 'future'),
        'max_episode_length': maximum_episode_length
    }

    # Add action noise for better exploration
    action_noise = NormalActionNoise(
        mean=np.zeros(env.action_space.shape), 
        sigma=0.1 * np.ones(env.action_space.shape)
    )

    # Enhanced model
    model = SAC(
        MultiInputPolicy,
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=replay_buffer_kwargs,
        verbose=1,
        action_noise=action_noise,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        policy_kwargs=dict(net_arch=[512, 512, 256]),
        **sac_config
    )

    # Save configuration
    save_config_file(sac_config, env, os.path.join(log_relative_path, 'config.json'))

    # Enhanced callbacks
    checkpoint_callback = EnhancedCheckpointCallback(
        save_freq=validate_every_timesteps,
        save_path=log_dir
    )
    
    wandb_metrics_callback = WandbMetricsCallback(
        eval_env=eval_env,
        eval_freq=validate_every_timesteps // 4,  # Evaluate 4 times per validation
        verbose=2
    )
    
    curriculum_callback = CurriculumProgressCallback(
        curriculum_wrapper=env if hasattr(env, 'intervention_actors') else None,
        verbose=1
    )
    
    callbacks = []
    if wandb_config:
        wandb_callback = WandbCallback(
            gradient_save_freq=50000,
            model_save_path=f"models/{wandb.run.id}",
            verbose=2,
        )
    callbacks.append(wandb_callback)  # Add W&B callback first

    callbacks.extend([checkpoint_callback, wandb_metrics_callback, curriculum_callback])

    # Log initial environment info
    if wandb_config:
        wandb.log({
            'env/action_space_shape': env.action_space.shape,
            'env/observation_space_keys': list(env.observation_space.spaces.keys()),
            'env/max_episode_length': maximum_episode_length,
            'model/total_parameters': sum(p.numel() for p in model.policy.parameters()),
            'model/device': str(model.device)
        })

    # Training loop
    start_time = time.time()
    for i in range(int(total_time_steps / validate_every_timesteps)):
        print(f"\n=== Training Iteration {i+1}/{int(total_time_steps / validate_every_timesteps)} ===")
        
        iteration_start = time.time()
        model.learn(
            total_timesteps=validate_every_timesteps,
            tb_log_name="enhanced_sac_her",
            reset_num_timesteps=False,
            callback=callback
        )
        
        iteration_time = time.time() - iteration_start
        total_time = time.time() - start_time
        
        # Log training progress
        if wandb_config:
            wandb.log({
                'training/iteration': i+1,
                'training/iteration_time_minutes': iteration_time / 60,
                'training/total_time_hours': total_time / 3600,
                'training/timesteps_per_second': validate_every_timesteps / iteration_time,
                'train/timesteps': (i+1) * validate_every_timesteps
            })
        
        model.save(os.path.join(log_relative_path, f'saved_model_{(i+1)*validate_every_timesteps}'))

    # Final model save
    final_model_path = os.path.join(log_relative_path, 'final_model')
    model.save(final_model_path)
    
    # Log final model as artifact
    if wandb_config:
        final_artifact = wandb.Artifact(
            name="final_model",
            type="model",
            description="Final trained SAC+HER model"
        )
        final_artifact.add_file(f"{final_model_path}.zip")
        wandb.log_artifact(final_artifact)
        
        wandb.finish()
    
    print("Training complete. Final model saved.")

    # Cleanup
    env.close()
    eval_env.close()
    
    return model


# --- Evaluation Function ---

def evaluate_trained_model(model_path, log_path):
    """Evaluate the trained model using the CausalWorld reaching benchmark."""
    print("Loading trained model for evaluation...")
    model = SAC.load(os.path.join(model_path, 'final_model'))
    
    def policy_fn(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action
    
    # Use the reaching benchmark for comprehensive evaluation
    evaluation_protocols = REACHING_BENCHMARK['evaluation_protocols']
    
    evaluator = EvaluationPipeline(
        evaluation_protocols=evaluation_protocols,
        tracker_path=log_path,
        initial_seed=42
    )
    
    # Evaluate on full benchmark
    print("Running benchmark evaluation...")
    scores = evaluator.evaluate_policy(policy_fn, fraction=1.0)
    evaluator.save_scores(log_path)
    
    print("Evaluation Results:")
    for protocol_name, score in scores.items():
        print(f"  {protocol_name}: {score:.4f}")
    
    return scores


# --- Utility Functions ---

def save_config_file(sac_config, env, file_path):
    """Save the configuration of the task, environment, and SAC to a JSON file."""
    task_config = env.get_task().get_task_params()
    for task_param in task_config:
        if not isinstance(task_config[task_param], str):
            task_config[task_param] = str(task_config[task_param])
    env_config = env.get_world_params()
    configs_to_save = [task_config, env_config, sac_config]
    with open(file_path, 'w') as fout:
        json.dump(configs_to_save, fout, indent=2)


# --- Main Execution ---

if __name__ == '__main__':
    # ----------------------
    # Argument Parsing
    # ----------------------
    parser = argparse.ArgumentParser(description='Enhanced HER+SAC Training with Curriculum and W&B')
    parser.add_argument("--seed_num", default=42, type=int, help="Random seed for reproducibility")
    parser.add_argument("--skip_frame", default=5, type=int, help="Frame skipping for faster training")
    parser.add_argument("--max_episode_length", default=1500, type=int, help="Maximum steps per episode")  # Optimized length
    parser.add_argument("--total_time_steps_per_update", default=250000, type=int, help="Timesteps per validation/checkpoint")
    parser.add_argument("--task_name", default="reaching", help="CausalWorld task name")
    parser.add_argument("--log_relative_path", default="./enhanced_reaching_logs", help="Where to save logs and models")
    parser.add_argument("--evaluate_only", action='store_true', help="Only run evaluation")
    parser.add_argument("--total_timesteps", default=5000000, type=int, help="Total training timesteps")  # 5M total steps
    parser.add_argument("--wandb_project", default="causalworld/sac_her_reaching/simple_curriculum", help="WandB project name")
    parser.add_argument("--wandb_run_name", default=None, help="WandB run name (optional)")
    args = parser.parse_args()

    # ----------------------
    # W&B Configuration
    # ----------------------
    # Weights & Biases is used for experiment tracking, logging, and artifact management.
    # The run name is timestamped for uniqueness if not provided.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb_config = {
        'project': args.wandb_project,
        'run_name': args.wandb_run_name or f"reaching_seed{args.seed_num}_{timestamp}",
    }

    # ----------------------
    # Enhanced SAC configuration for better reward accrual
    # ----------------------
    # These hyperparameters are tuned for CausalWorld's reaching task.
    sac_config = {
        "n_sampled_goal": 4,
        "goal_selection_strategy": 'future',
        "learning_rate": 3e-4,
        "train_freq": (1, "step"),
        "gradient_steps": 1,
        "learning_starts": 5000,  # Earlier start
        # Use W&B run name for tensorboard log if W&B is enabled, else fallback to log path
        "tensorboard_log": f"runs/{wandb_config['run_name']}" if wandb_config else args.log_relative_path,
        "buffer_size": int(1e6),
        "gamma": 0.98,
        "batch_size": 256,
        "ent_coef": 'auto',
        "target_update_interval": 1,
        "tau": 0.005,
        "target_entropy": 'auto'
    }

    # ----------------------
    # Main Execution Logic
    # ----------------------
    # If --evaluate_only is set, skip training and only run evaluation.
    if args.evaluate_only:
        # Run evaluation only
        scores = evaluate_trained_model(args.log_relative_path, args.log_relative_path)
    else:
        # Run training
        train_enhanced_policy(
            log_relative_path=args.log_relative_path,
            maximum_episode_length=args.max_episode_length,
            skip_frame=args.skip_frame,
            seed_num=args.seed_num,
            sac_config=sac_config,
            total_time_steps=args.total_timesteps,  # Use the argument value
            validate_every_timesteps=args.total_time_steps_per_update,
            task_name=args.task_name,
            wandb_config=wandb_config
        )

        print("Training completed successfully!")
        
        # Run evaluation after training
        print("\n" + "="*50)
        print("Training completed. Starting evaluation...")
        scores = evaluate_trained_model(args.log_relative_path, args.log_relative_path)


## TMUX commands for long sessions

'''
# Start a new tmux session
tmux new-session -d -s causal_training

# Send the training command to the session
tmux send-keys -t causal_training "cd ~/causal-core-su25" Enter
tmux send-keys -t causal_training "conda activate causal_env" Enter
tmux send-keys -t causal_training "python her_sac_curriculum.py --log_relative_path ./enhanced_reaching_logs --total_time_steps_per_update 250000 --total_timesteps 5000000 --seed_num 42" Enter

# Detach from session (training continues in background)
tmux detach-session -t causal_training

# To reattach later:
# tmux attach-session -t causal_training

# To check running sessions:
# tmux list-sessions

# To kill the session when done:
# tmux kill-session -t causal_training
'''
