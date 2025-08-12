import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import logging
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, BaseCallback
import wandb
from wandb.integration.sb3 import WandbCallback
from causal_world.task_generators import generate_task
from causal_world.envs import CausalWorld
from causal_world.intervention_actors import (
    GoalInterventionActorPolicy, 
    PhysicalPropertiesInterventionActorPolicy,
    VisualInterventionActorPolicy,
    JointsInterventionActorPolicy,
    RigidPoseInterventionActorPolicy,
    RandomInterventionActorPolicy
)
import multiprocessing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(process)d %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler("adaptive_curriculum_on_ppo_training_debug.log"),
        logging.StreamHandler()
    ]
)

DENSE_REWARD_WEIGHTS = {
    'pushing': [750, 250, 100],
    'picking': [250, 0, 125, 0, 750, 0, 0, 0.005],
    'reaching': [100000, 0, 0, 0],
    'pick_and_place': [750, 50, 250, 0, 0.005],
    'stacking2': [750, 250, 250, 125, 0.005],
}

class AdaptiveCurriculumManager:
    def __init__(self, interventions, window_size=20, reward_threshold=0.01, plateau_patience=10, shared_idx=None):
        self.interventions = interventions
        if shared_idx is None:
            self.current_idx = multiprocessing.Value('i', 0)
        else:
            self.current_idx = shared_idx
        self.window_size = window_size
        self.reward_threshold = reward_threshold
        self.plateau_patience = plateau_patience
        self.recent_rewards = []
        self.best_mean_reward = -np.inf
        self.plateau_counter = 0
        self.episode_counter = 0
        print(f"[ADAPTIVE] Initialized with {len(interventions)} interventions.")

    def record_reward(self, reward):
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > self.window_size:
            self.recent_rewards.pop(0)
        self.episode_counter += 1

    def should_advance(self):
        # If curriculum is complete, do not advance or increment plateau
        if self.current_idx.value >= len(self.interventions) - 1:
            return False
        if len(self.recent_rewards) < self.window_size:
            return False
        mean_reward = np.mean(self.recent_rewards)
        print(f"[ADAPTIVE] Mean reward over last {self.window_size} episodes: {mean_reward:.6f}")
        if mean_reward > self.best_mean_reward + self.reward_threshold:
            self.best_mean_reward = mean_reward
            self.plateau_counter = 0
            return False
        else:
            self.plateau_counter += 1
            print(f"[ADAPTIVE] Plateau counter: {self.plateau_counter}/{self.plateau_patience}")
            if self.plateau_counter >= self.plateau_patience:
                return True
        return False

    def advance(self):
        if self.current_idx.value < len(self.interventions) - 1:
            self.current_idx.value += 1
            print(f"[ADAPTIVE] Advancing to intervention {self.current_idx.value}: {type(self.interventions[self.current_idx.value]).__name__}")
            self.plateau_counter = 0
            self.best_mean_reward = -np.inf
            self.recent_rewards = []
        else:
            print("[ADAPTIVE] Curriculum complete.")
            # No further advancement or plateau logging after this

    def get_current_intervention(self):
        return self.interventions[self.current_idx.value]

class AdaptiveCurriculumWrapper:
    def __init__(self, base_env, curriculum_manager):
        self.base_env = base_env
        self.curriculum_manager = curriculum_manager
        self.episode_count = 0

    def reset(self):
        obs = self.base_env.reset()
        self.episode_count += 1
        # Apply current intervention
        actor = self.curriculum_manager.get_current_intervention()
        print(f"[DEBUG] Environment reset: current_idx={self.curriculum_manager.current_idx.value}, actor={type(actor).__name__}")
        intervention_dict = actor._act(self.base_env.get_variable_space_used())
        if intervention_dict:
            self.base_env.do_intervention(intervention_dict)
            print(f"[ADAPTIVE] Applied intervention: {type(actor).__name__}")
        return obs

    def step(self, action):
        return self.base_env.step(action)

    def __getattr__(self, name):
        return getattr(self.base_env, name)

class AdaptiveCurriculumCallback(BaseCallback):
    def __init__(self, curriculum_manager, env, wandb_enabled=False, verbose=0):
        super().__init__(verbose)
        self.curriculum_manager = curriculum_manager
        self.env = env
        self.wandb_enabled = wandb_enabled
        self.episode_rewards = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        if self.locals.get("dones") is not None and self.locals["dones"][0]:
            reward = self.locals["rewards"][0]
            self.episode_rewards.append(reward)
            self.curriculum_manager.record_reward(reward)
            self.episode_count += 1
            print(f"[ADAPTIVE] Episode {self.episode_count} reward: {reward:.6f}")
            if self.wandb_enabled:
                wandb.log({
                    'adaptive/episode_reward': reward,
                    'adaptive/mean_reward': np.mean(self.curriculum_manager.recent_rewards),
                    'adaptive/current_intervention': type(self.curriculum_manager.get_current_intervention()).__name__,
                    'adaptive/intervention_idx': self.curriculum_manager.current_idx.value,
                    'adaptive/episode': self.episode_count
                }, step=self.num_timesteps)
            if self.curriculum_manager.should_advance():
                self.curriculum_manager.advance()
                print(f"[ADAPTIVE] Curriculum advanced to intervention {self.curriculum_manager.current_idx.value}")
                if self.wandb_enabled:
                    wandb.log({
                        'adaptive/curriculum_advanced': self.curriculum_manager.current_idx.value,
                        'adaptive/episode': self.episode_count
                    }, step=self.num_timesteps)
        return True

def make_adaptive_env(rank, task_name, curriculum_manager, seed=0, skip_frame=3, max_episode_length=250):
    def _init():
        dense_weights = DENSE_REWARD_WEIGHTS.get(task_name, [0])
        task = generate_task(
            task_generator_id=task_name,
            dense_reward_weights=np.array(dense_weights),
            variables_space='space_a',
            fractional_reward_weight=1
        )
        base_env = CausalWorld(
            task=task,
            skip_frame=skip_frame,
            action_mode='joint_torques',
            enable_visualization=False,
            seed=seed + rank,
            max_episode_length=max_episode_length
        )
        # Initialize all intervention actors with the environment
        for actor in curriculum_manager.interventions:
            if hasattr(actor, "initialize"):
                actor.initialize(base_env)
        env = AdaptiveCurriculumWrapper(base_env, curriculum_manager)
        return env
    return _init

def main():
    parser = argparse.ArgumentParser(description='PPO with Adaptive Curriculum for CausalWorld')
    parser.add_argument('--train', action='store_true', help='Train with adaptive curriculum wrapper')
    parser.add_argument('--task', type=str, default='pushing', 
                       choices=list(DENSE_REWARD_WEIGHTS.keys()), help='Task name')
    parser.add_argument('--pretrained_path', type=str, 
                       help='Path to pretrained PPO model')
    parser.add_argument('--log_dir', type=str, default='adaptive_curriculum_on_ppo', help='Log directory')
    parser.add_argument('--timesteps', type=int, default=3_000_000, help='Total timesteps')
    parser.add_argument('--num_envs', type=int, default=16, help='Number of parallel environments')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--skip_frame', type=int, default=3, help='Frame skip')
    parser.add_argument('--max_episode_length', type=int, default=250, help='Max episode length')
    parser.add_argument('--eval_freq', type=int, default=100_000, help='Evaluation frequency')
    parser.add_argument('--use_wandb', action='store_true', help='Enable wandb logging')
    args = parser.parse_args()

    logging.info(f"[ARGS] {args}")

    if args.pretrained_path is None:
        args.pretrained_path = f'ppo_{args.task}_sb3/final_model.zip'
        logging.info(f"[PRETRAINED] Using pretrained model path: {args.pretrained_path}")

    ppo_config = {
        'gamma': 0.995,
        'n_steps': 4096 // args.num_envs,
        'ent_coef': 0.02,
        'learning_rate': 2.5e-4,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'gae_lambda': 0.97,
        'batch_size': 512,
        'n_epochs': 15
    }

    wandb_config = None
    if args.use_wandb:
        wandb_config = {
            'project': f'adaptive-curriculum-ppo-{args.task}',
            'run_name': f'adaptive_curriculum_ppo_{args.task}_seed{args.seed}'
        }
        wandb.init(
            project=wandb_config['project'],
            name=wandb_config['run_name'],
            config={
                'task_name': args.task,
                'curriculum_type': 'Adaptive_Curriculum_on_PPO',
                'max_episode_length': args.max_episode_length,
                'skip_frame': args.skip_frame,
                'seed': args.seed,
                'total_timesteps': args.timesteps,
                **ppo_config
            },
            tags=[args.task, 'PPO', 'Adaptive_Curriculum_on_PPO', 'curriculum', 'adaptive'],
            sync_tensorboard=True
        )
        logging.info(f"[WANDB] Initialized with project: {wandb_config['project']} and run name: {wandb_config['run_name']}")

    if not args.log_dir.startswith('adaptive_curriculum_on_ppo'):
        args.log_dir = f'adaptive_curriculum_on_ppo_{args.log_dir}'
    os.makedirs(args.log_dir, exist_ok=True)
    logging.info(f"[LOGDIR] Log directory: {args.log_dir}")

    # Define interventions for curriculum
    # TODO: you need to fix the joints intervention; the values are going out of bounds
    interventions = [
        GoalInterventionActorPolicy(),
        PhysicalPropertiesInterventionActorPolicy(group='tool'),
        VisualInterventionActorPolicy(),
        JointsInterventionActorPolicy(),
        RigidPoseInterventionActorPolicy(positions=True, orientations=True),
        RandomInterventionActorPolicy()
    ]
    logging.info(f"[CURRICULUM] Interventions: {[type(i).__name__ for i in interventions]}")

    # Create a shared value for current_idx
    manager = multiprocessing.Manager()
    shared_idx = manager.Value('i', 0)
    curriculum_manager = AdaptiveCurriculumManager(interventions, shared_idx=shared_idx)

    # Create vectorized environment
    logging.info(f"[ENV] Creating {args.num_envs} parallel training environments for task: {args.task}")
    def make_env_with_shared_idx(rank, task_name, curriculum_manager, seed=0, skip_frame=3, max_episode_length=250, shared_idx=None):
        def _init():
            dense_weights = DENSE_REWARD_WEIGHTS.get(task_name, [0])
            task = generate_task(
                task_generator_id=task_name,
                dense_reward_weights=np.array(dense_weights),
                variables_space='space_a',
                fractional_reward_weight=1
            )
            base_env = CausalWorld(
                task=task,
                skip_frame=skip_frame,
                action_mode='joint_torques',
                enable_visualization=False,
                seed=seed + rank,
                max_episode_length=max_episode_length
            )
            # Initialize all intervention actors with the environment
            for actor in curriculum_manager.interventions:
                if hasattr(actor, "initialize"):
                    actor.initialize(base_env)
            # Pass the curriculum_manager with shared_idx
            env = AdaptiveCurriculumWrapper(base_env, AdaptiveCurriculumManager(curriculum_manager.interventions, shared_idx=shared_idx))
            return env
        return _init

    env = SubprocVecEnv([
        make_env_with_shared_idx(i, args.task, curriculum_manager, args.seed, args.skip_frame, args.max_episode_length, shared_idx)
        for i in range(args.num_envs)
    ])
    env = VecMonitor(env, filename=os.path.join(args.log_dir, 'monitor.csv'))

    # Create evaluation environment (without curriculum for consistent evaluation)
    logging.info(f"[ENV] Creating evaluation environment for task: {args.task}")
    def make_eval_env():
        dense_weights = DENSE_REWARD_WEIGHTS.get(args.task, [0])
        task = generate_task(
            task_generator_id=args.task,
            dense_reward_weights=np.array(dense_weights),
            variables_space='space_a',
            fractional_reward_weight=1
        )
        return CausalWorld(
            task=task,
            skip_frame=args.skip_frame,
            action_mode='joint_torques',
            enable_visualization=False,
            seed=args.seed,
            max_episode_length=args.max_episode_length
        )
    eval_env = make_eval_env()

    # Load pretrained model
    logging.info(f"[MODEL] PPO model loaded from: {args.pretrained_path}")
    policy_kwargs = dict(activation_fn=nn.LeakyReLU, net_arch=[512, 256])
    model = PPO.load(
        args.pretrained_path,
        env=env,
        policy_kwargs=policy_kwargs,
        tensorboard_log=args.log_dir,
        seed=args.seed,
        **ppo_config
    )

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.eval_freq // args.num_envs, 1),
        save_path=os.path.join(args.log_dir, 'logs'),
        name_prefix='adaptive_curriculum_on_ppo'
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.log_dir, 'logs', 'best_model'),
        log_path=os.path.join(args.log_dir, 'logs'),
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False,
        verbose=1
    )
    adaptive_callback = AdaptiveCurriculumCallback(curriculum_manager, env, wandb_enabled=args.use_wandb)
    callbacks = [checkpoint_callback, eval_callback, adaptive_callback]
    if wandb_config:
        wandb_callback = WandbCallback(
            gradient_save_freq=10000,
            model_save_path=os.path.join(args.log_dir, 'wandb_models'),
            verbose=2
        )
        callbacks.append(wandb_callback)
    callback = CallbackList(callbacks)
    logging.info(f"[CALLBACKS] Callbacks set up: {[type(cb).__name__ for cb in callbacks]}")

    # Train model
    logging.info(f"[TRAIN] Starting training for {args.timesteps} timesteps...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        tb_log_name="adaptive_curriculum_on_ppo"
    )
    logging.info(f"[TRAIN] Training complete.")

    # Save final model
    model.save(os.path.join(args.log_dir, 'adaptive_curriculum_on_ppo'))
    logging.info(f"[SAVE] Model saved to: {os.path.join(args.log_dir, 'adaptive_curriculum_on_ppo')}")
    if wandb_config:
        wandb.finish()
    logging.info(f"Adaptive Curriculum on PPO training completed. Model saved to {args.log_dir}")

if __name__ == '__main__':
    main() 