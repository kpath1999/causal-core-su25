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
from causal_world.wrappers.curriculum_wrappers import CurriculumWrapper
from causal_world.intervention_actors import (
    GoalInterventionActorPolicy, 
    PhysicalPropertiesInterventionActorPolicy,
    VisualInterventionActorPolicy,
    JointsInterventionActorPolicy,
    RigidPoseInterventionActorPolicy,
    RandomInterventionActorPolicy
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(process)d %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler("simple_curriculum_on_ppo_training_debug.log"),
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

def create_curriculum_interventions(task_name):
    # You can customize these for each task if needed
    interventions = [
        GoalInterventionActorPolicy(),
        PhysicalPropertiesInterventionActorPolicy(group='tool'),
        VisualInterventionActorPolicy(),
        JointsInterventionActorPolicy(),
        RigidPoseInterventionActorPolicy(positions=True, orientations=True),
        RandomInterventionActorPolicy()
    ]
    # (episode_start, episode_end, episode_periodicity, intervention_idx)
    # Here, we use a simple schedule: each intervention for 100k episodes, sequentially
    curriculum_schedule = [
        (0, 100000, 1, 0),
        (100000, 200000, 1, 1),
        (200000, 300000, 1, 2),
        (300000, 400000, 1, 3),
        (400000, 500000, 1, 4),
        (500000, 1000000, 1, 5)
    ]
    logging.info(f"[CURRICULUM] Intervention schedule:")
    for idx, (start, end, period, actor_idx) in enumerate(curriculum_schedule):
        logging.info(f"  - Actor {actor_idx} ({type(interventions[actor_idx]).__name__}): episodes {start} to {end}, every {period} episode(s)")
    return interventions, curriculum_schedule

def make_curriculum_env(rank, task_name, seed=0, skip_frame=3, max_episode_length=250):
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
        interventions, schedule = create_curriculum_interventions(task_name)
        env = CurriculumWrapper(
            base_env,
            intervention_actors=interventions,
            actives=schedule
        )
        return env
    return _init

def main():
    parser = argparse.ArgumentParser(description='PPO with Simple Curriculum for CausalWorld')
    parser.add_argument('--train', action='store_true', help='Train with curriculum wrapper')
    parser.add_argument('--task', type=str, default='pushing', 
                       choices=list(DENSE_REWARD_WEIGHTS.keys()), help='Task name')
    parser.add_argument('--pretrained_path', type=str, 
                       help='Path to pretrained PPO model')
    parser.add_argument('--log_dir', type=str, default='simple_curriculum_on_ppo', help='Log directory')
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
            'project': f'simple-curriculum-ppo-{args.task}',
            'run_name': f'simple_curriculum_ppo_{args.task}_seed{args.seed}'
        }
        wandb.init(
            project=wandb_config['project'],
            name=wandb_config['run_name'],
            config={
                'task_name': args.task,
                'curriculum_type': 'Simple_Curriculum_on_PPO',
                'max_episode_length': args.max_episode_length,
                'skip_frame': args.skip_frame,
                'seed': args.seed,
                'total_timesteps': args.timesteps,
                **ppo_config
            },
            tags=[args.task, 'PPO', 'Simple_Curriculum_on_PPO', 'curriculum', 'wrapper'],
            sync_tensorboard=True
        )
        logging.info(f"[WANDB] Initialized with project: {wandb_config['project']} and run name: {wandb_config['run_name']}")

    if not args.log_dir.startswith('simple_curriculum_on_ppo'):
        args.log_dir = f'simple_curriculum_on_ppo_{args.log_dir}'
    os.makedirs(args.log_dir, exist_ok=True)
    logging.info(f"[LOGDIR] Log directory: {args.log_dir}")

    # Create vectorized environment
    logging.info(f"[ENV] Creating {args.num_envs} parallel training environments for task: {args.task}")
    env = SubprocVecEnv([
        make_curriculum_env(i, args.task, args.seed, args.skip_frame, args.max_episode_length)
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
        name_prefix='simple_curriculum_on_ppo'
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
    callbacks = [checkpoint_callback, eval_callback]
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
        tb_log_name="simple_curriculum_on_ppo"
    )
    logging.info(f"[TRAIN] Training complete.")

    # Save final model
    model.save(os.path.join(args.log_dir, 'simple_curriculum_on_ppo'))
    logging.info(f"[SAVE] Model saved to: {os.path.join(args.log_dir, 'simple_curriculum_on_ppo')}")
    if wandb_config:
        wandb.finish()
    logging.info(f"Simple Curriculum on PPO training completed. Model saved to {args.log_dir}")

if __name__ == '__main__':
    main() 