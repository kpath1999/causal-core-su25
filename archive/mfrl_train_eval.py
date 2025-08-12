##
# Usage Instructions
# Training Only:

# bash
# python your_script.py --train --timesteps=60000000
# Evaluation Only (after training):

# bash
# python your_script.py --eval --no-train
# Both Training and Evaluation:

# bash
# python your_script.py --train --eval --timesteps=60000000
##

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Before other imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
from absl import app, flags

from causal_world.task_generators.task import generate_task
from causal_world.envs.causalworld import CausalWorld
from causal_world.evaluation import EvaluationPipeline
from causal_world.benchmark import PICKING_BENCHMARK
import causal_world.evaluation.visualization.visualiser as vis

from stable_baselines3 import SAC
from stable_baselines3.sac import MultiInputPolicy
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from causal_world.wrappers.env_wrappers import HERGoalEnvWrapper

import wandb
from wandb.integration.sb3 import WandbCallback

FLAGS = flags.FLAGS
flags.DEFINE_bool('train', True, 'Train the model.')
flags.DEFINE_bool('eval', True, 'Evaluate the model.')
flags.DEFINE_float('timesteps', 60e6, 'Total timesteps.')
flags.DEFINE_string('task', 'picking', 'Task selected.')

class CustomCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=1):
        super(CustomCheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f"rl_model_{self.num_timesteps}")
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Saving model checkpoint to {model_path}")
        return True

class HEREvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq, log_path=None, verbose=1):
        super(HEREvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.log_path = log_path
        self.best_success_rate = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            episode_rewards = []
            success_count = 0
            obs = self.eval_env.reset()
            
            for _ in range(20):  # Run 20 episodes for better statistics
                done = False
                total_reward = 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, info = self.eval_env.step(action)
                    total_reward += reward
                
                episode_rewards.append(total_reward)
                # Check if task was successful (you may need to adjust this based on your task)
                if hasattr(info, 'is_success') and info['is_success']:
                    success_count += 1
                obs = self.eval_env.reset()

            mean_reward = np.mean(episode_rewards)
            success_rate = success_count / 20.0
            
            if self.verbose > 0:
                print(f"[Eval] Step: {self.num_timesteps}, Mean Reward: {mean_reward:.2f}, Success Rate: {success_rate:.2f}")

            # Save best model based on success rate
            if success_rate > self.best_success_rate:
                self.best_success_rate = success_rate
                best_path = os.path.join(self.log_path or ".", "best_model")
                os.makedirs(best_path, exist_ok=True)
                self.model.save(os.path.join(best_path, "best_model"))
                if self.verbose > 0:
                    print(f"New best model saved with success rate: {success_rate:.2f}")

        return True


def train_policy(num_of_envs, log_relative_path, maximum_episode_length,
                 skip_frame, seed_num, her_config, total_time_steps,
                 validate_every_timesteps, task_name, wandb_config=None):
    
    print("\n====== Training Setup ======")
    print(f"• Task: {task_name}")
    print(f"• Total timesteps: {total_time_steps:,}")
    print(f"• Episode length: {maximum_episode_length}")
    print(f"• Skip frame: {skip_frame}")
    print(f"• Seed: {seed_num}")
    
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
                **her_config
            },
            tags=[task_name, 'HER', 'SAC'],
            sync_tensorboard=True
        )
    
    # Create training environment
    task = generate_task(
        task_generator_id=task_name,
        dense_reward_weights=np.array([0]*8),
        fractional_reward_weight=1,
        goal_height=0.15,
        tool_block_mass=0.02
    )

    env = CausalWorld(
        task=task,
        skip_frame=skip_frame,
        enable_visualization=False,
        seed=seed_num,
        max_episode_length=maximum_episode_length
    )
    
    # Apply HER wrapper and monitoring
    env = HERGoalEnvWrapper(env)
    env = Monitor(env, log_relative_path)
    
    # Create evaluation environment
    eval_task = generate_task(
        task_generator_id=task_name,
        dense_reward_weights=np.array([0]*8),
        fractional_reward_weight=1,
        goal_height=0.15,
        tool_block_mass=0.02
    )
    eval_env = CausalWorld(
        task=eval_task,
        skip_frame=skip_frame,
        enable_visualization=False,
        seed=seed_num + 1000,
        max_episode_length=maximum_episode_length
    )
    eval_env = HERGoalEnvWrapper(eval_env)
    
    set_random_seed(seed_num)
    
    # Setup callbacks
    checkpoint_callback = CustomCheckpointCallback(
        save_freq=validate_every_timesteps,
        save_path=os.path.join(log_relative_path, 'checkpoints'),
        verbose=1
    )
    
    eval_callback = HEREvalCallback(
        eval_env=eval_env,
        eval_freq=validate_every_timesteps,
        log_path=log_relative_path,
        verbose=1
    )
    
    callbacks = [checkpoint_callback, eval_callback]
    
    if wandb_config:
        wandb_callback = WandbCallback(
            gradient_save_freq=10000,
            model_save_path=f"models/{wandb.run.id}",
            verbose=2,
        )
        callbacks.append(wandb_callback)
    
    callback_list = CallbackList(callbacks)
    
    # Extract HER parameters
    replay_buffer_kwargs = {
        'n_sampled_goal': her_config.pop('n_sampled_goal', 4),
        'goal_selection_strategy': her_config.pop('goal_selection_strategy', 'future'),
        'max_episode_length': maximum_episode_length
    }
    
    # Create SAC model with HER
    model = SAC(
        MultiInputPolicy,
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=replay_buffer_kwargs,
        verbose=1,
        tensorboard_log=her_config.pop('tensorboard_log', log_relative_path),
        **her_config,
        seed=seed_num
    )
    
    print("\n====== Model Architecture ======")
    print(f"• Policy: {model.policy}")
    print(f"• Replay Buffer: HER with {replay_buffer_kwargs['n_sampled_goal']} sampled goals")
    print(f"• Goal Selection: {replay_buffer_kwargs['goal_selection_strategy']}")
    
    # Check for existing checkpoints
    checkpoint_dir = os.path.join(log_relative_path, 'checkpoints')
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('rl_model_')]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split('_')[-1]))
            latest_checkpoint = checkpoints[-1]
            checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
            print(f"\nLoading checkpoint: {checkpoint_path}")
            model.load(checkpoint_path)
            done_timesteps = int(latest_checkpoint.split('_')[-1])
            print(f"Resuming from {done_timesteps:,} timesteps")
        else:
            done_timesteps = 0
    else:
        done_timesteps = 0
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    print("\n====== Starting Training ======")
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_time_steps - done_timesteps,
            callback=callback_list,
            tb_log_name="her_sac",
            reset_num_timesteps=False if done_timesteps > 0 else True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving current model...")
        model.save(os.path.join(log_relative_path, 'interrupted_model'))
        print("Model saved successfully.")
        return model
    
    training_time = (time.time() - start_time) / 3600
    print(f"\nTraining completed in {training_time:.2f} hours")
    
    # Save final model
    final_model_path = os.path.join(log_relative_path, 'final_model')
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    if wandb_config:
        wandb.finish()
    
    # Clean up environments
    env.close()
    eval_env.close()
    
    return model

def evaluate_training_progress(log_dir):
    """Evaluate training progress from logs"""
    print("\n====== Training Progress Analysis ======")
    
    monitor_file = os.path.join(log_dir, 'monitor.csv')
    if not os.path.exists(monitor_file):
        print(f"No monitor file found at {monitor_file}")
        return
    
    # Load training data
    df = pd.read_csv(monitor_file, skiprows=1)
    
    print(f"• Total episodes: {len(df):,}")
    print(f"• Final mean reward: {df['r'].rolling(100).mean().iloc[-1]:.2f}")
    print(f"• Best mean reward: {df['r'].rolling(100).mean().max():.2f}")
    print(f"• Mean episode length: {df['l'].mean():.1f}")
    
    # Plot training progress
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(df['r'].rolling(100).mean())
    plt.title('Episode Reward (100-episode moving average)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(df['l'].rolling(100).mean())
    plt.title('Episode Length (100-episode moving average)')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(np.cumsum(df['l']))
    plt.title('Cumulative Timesteps')
    plt.xlabel('Episode')
    plt.ylabel('Total Steps')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'training_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"Training analysis saved to: {os.path.join(log_dir, 'training_analysis.png')}")
    plt.show()

def benchmark_evaluation(log_dir, task_name='picking'):
    """Run comprehensive benchmark evaluation using CausalWorld's evaluation pipeline"""
    print("\n====== Benchmark Evaluation ======")
    
    # Find best model
    best_model_path = os.path.join(log_dir, 'best_model', 'best_model.zip')
    if not os.path.exists(best_model_path):
        # Fallback to final model
        best_model_path = os.path.join(log_dir, 'final_model.zip')
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"No trained model found in {log_dir}")
    
    print(f"Loading model from: {best_model_path}")
    model = SAC.load(best_model_path)
    
    # Setup evaluation pipeline
    task_params = {'task_generator_id': task_name}
    world_params = {
        'skip_frame': 3,
        'max_episode_length': 600
    }
    
    # Use the appropriate benchmark for picking task
    if task_name == 'picking':
        evaluation_protocols = PICKING_BENCHMARK['evaluation_protocols']
    else:
        # You can add other benchmarks here
        raise ValueError(f"No benchmark available for task: {task_name}")
    
    evaluator = EvaluationPipeline(
        evaluation_protocols=evaluation_protocols,
        task_params=task_params,
        world_params=world_params,
        visualize_evaluation=False
    )
    
    def policy_fn(obs):
        """Policy function for evaluation"""
        action, _ = model.predict(obs, deterministic=True)
        return action
    
    print("Running benchmark evaluation...")
    print("This may take several minutes depending on the benchmark complexity...")
    
    # Run evaluation with a reasonable fraction for testing
    scores = evaluator.evaluate_policy(policy_fn, fraction=0.01)  # Use 1% for faster evaluation
    
    print("\n====== Evaluation Results ======")
    for metric, value in scores.items():
        if isinstance(value, (int, float)):
            print(f"• {metric.replace('_', ' ').title()}: {value:.4f}")
        else:
            print(f"• {metric.replace('_', ' ').title()}: {value}")
    
    # Generate visual analysis
    experiments = {f'HER-SAC ({task_name})': scores}
    vis.generate_visual_analysis(log_dir, experiments=experiments)
    print(f"\nVisual analysis saved to: {log_dir}")
    
    return scores

def main(argv):
    del argv
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_relative_path = f'her_sac_{FLAGS.task}_{timestamp}'
    
    print("\n====== Experiment Configuration ======")
    print(f"• Task: {FLAGS.task}")
    print(f"• Total timesteps: {FLAGS.timesteps:,}")
    print(f"• Log directory: {log_relative_path}")
    
    # Training parameters
    training_params = {
        'num_of_envs': 1,  # HER typically works with single environment
        'log_relative_path': log_relative_path,
        'maximum_episode_length': 600,
        'skip_frame': 3,
        'seed_num': 0,
        'total_time_steps': int(FLAGS.timesteps),
        'validate_every_timesteps': 100000,  # Evaluate every 100k steps
        'task_name': FLAGS.task
    }
    
    # HER and SAC hyperparameters
    her_config = {
        "n_sampled_goal": 4,
        "goal_selection_strategy": 'future',
        "gamma": 0.98,
        "tau": 0.01,
        "ent_coef": 'auto',
        "target_entropy": -9,
        "learning_rate": 0.00025,
        "buffer_size": 1000000,
        "learning_starts": 1000,
        "batch_size": 256,
        "train_freq": 1,
        "gradient_steps": 1,
        "tensorboard_log": log_relative_path
    }
    
    # W&B configuration
    wandb_config = {
        'project': 'causal-world-her-sac',
        'run_name': f'{FLAGS.task}_her_sac_{timestamp}',
        'tags': ['her', 'sac', FLAGS.task]
    }
    
    if FLAGS.train:
        print("\n=== Training Mode Activated ===")
        model = train_policy(
            her_config=her_config,
            wandb_config=wandb_config,
            **training_params
        )
    
    if FLAGS.eval:
        print("\n=== Evaluation Mode Activated ===")
        evaluate_training_progress(log_relative_path)
        benchmark_evaluation(log_relative_path, FLAGS.task)

if __name__ == '__main__':
    app.run(main)