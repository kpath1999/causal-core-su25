import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
import time  # Added for timing measurements
import gym
print("Gym version:", gym.__version__)

from causal_world.task_generators import generate_task
from causal_world.envs import CausalWorld
from causal_world.evaluation import EvaluationPipeline
from causal_world.benchmark import REACHING_BENCHMARK, PUSHING_BENCHMARK, PICKING_BENCHMARK, PICK_AND_PLACE_BENCHMARK, STACKING2_BENCHMARK
import causal_world.evaluation.visualization.visualiser as vis

import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, BaseCallback
import argparse

# Mapping from task name to benchmark
TASK_BENCHMARKS = {
    'reaching': REACHING_BENCHMARK,
    'pushing': PUSHING_BENCHMARK,
    'picking': PICKING_BENCHMARK,
    'pick_and_place': PICK_AND_PLACE_BENCHMARK,
    'stacking2': STACKING2_BENCHMARK,
    # Add more as available
}
SUPPORTED_TASKS = list(TASK_BENCHMARKS.keys())

# Mapping from task name to dense_reward_weights
DENSE_REWARD_WEIGHTS = {
    'reaching': [100000, 0, 0, 0],
    'pushing': [750, 250, 100],
    'picking': [250, 0, 125, 0, 750, 0, 0, 0.005],
    'pick_and_place': [750, 50, 250, 0, 0.005],
    'stacking2': [750, 250, 250, 125, 0.005],
}

# --- Custom callback to log CausalWorld metrics to wandb ---
class CausalWorldWandbCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=10000, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.best_success = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            episode_rewards = []
            success_rates = []
            for _ in range(5):
                obs = self.eval_env.reset()
                done = False
                total_reward = 0
                successes = 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, info = self.eval_env.step(action)
                    total_reward += reward
                    # CausalWorld envs often provide 'success' in info
                    if isinstance(info, dict) and 'success' in info:
                        successes += int(info['success'])
                episode_rewards.append(total_reward)
                success_rates.append(successes)
            mean_reward = np.mean(episode_rewards)
            mean_success = np.mean(success_rates)
            wandb.log({
                'eval/mean_reward': mean_reward,
                'eval/mean_success': mean_success,
                'timesteps': self.num_timesteps
            }, step=self.num_timesteps)
            if mean_success > self.best_success:
                self.best_success = mean_success
                self.model.save(os.path.join(self.model.logger.dir, 'best_success_model'))
        return True

# --- Environment factory ---
def make_env(rank, task_name, seed=0, skip_frame=3, max_episode_length=250, log_dir=None):
    def _init():
        dense_weights = DENSE_REWARD_WEIGHTS.get(task_name, [0])
        task = generate_task(
            task_generator_id=task_name,
            dense_reward_weights=np.array(dense_weights),
            variables_space='space_a',
            fractional_reward_weight=1
        )
        env = CausalWorld(
            task=task,
            skip_frame=skip_frame,
            action_mode='joint_torques',
            enable_visualization=False,
            seed=seed + rank,
            max_episode_length=max_episode_length
        )
        return env
    return _init

# --- Training function ---
def train_policy(num_envs, log_dir, max_episode_length, skip_frame, seed, ppo_config, total_timesteps, eval_freq, task_name, wandb_config=None):
    if wandb_config:
        wandb.init(
            project=wandb_config['project'],
            name=wandb_config['run_name'],
            config={
                'task_name': task_name,
                'max_episode_length': max_episode_length,
                'skip_frame': skip_frame,
                'seed': seed,
                'total_timesteps': total_timesteps,
                **ppo_config
            },
            tags=[task_name, 'PPO', 'SB3'],
            sync_tensorboard=True
        )

    os.makedirs(log_dir, exist_ok=True)
    env = SubprocVecEnv([
        make_env(i, task_name, seed, skip_frame, max_episode_length, log_dir=log_dir)
        for i in range(num_envs)
    ])
    env = VecMonitor(env, filename=os.path.join(log_dir, 'monitor.csv'))

    eval_env = make_env(0, task_name, seed, skip_frame, max_episode_length, log_dir=log_dir)()

    policy_kwargs = dict(activation_fn=nn.LeakyReLU, net_arch=[512, 256])
    model = PPO(
        MlpPolicy,
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
        **ppo_config,
        seed=seed
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(eval_freq // num_envs, 1),
        save_path=os.path.join(log_dir, 'logs'),
        name_prefix='ppo_model'
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_dir, 'logs', 'best_model'),
        log_path=os.path.join(log_dir, 'logs'),
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        verbose=1
    )
    callbacks = [checkpoint_callback, eval_callback]
    if wandb_config:
        wandb_callback = WandbCallback(
            gradient_save_freq=10000,
            model_save_path=os.path.join(log_dir, 'wandb_models'),
            verbose=2
        )
        cw_wandb_callback = CausalWorldWandbCallback(eval_env, eval_freq=eval_freq)
        callbacks.extend([wandb_callback, cw_wandb_callback])
    callback = CallbackList(callbacks)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        tb_log_name="ppo_sb3"
    )
    model.save(os.path.join(log_dir, 'final_model'))
    if wandb_config:
        wandb.finish()
    print(f"Final model saved to {os.path.join(log_dir, 'final_model')}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model.')
    parser.add_argument('--eval', action='store_true', help='Evaluate the model.')
    parser.add_argument('--timesteps', type=int, default=10_000_000, help='Total timesteps.')
    parser.add_argument('--num_envs', type=int, default=24, help='Number of parallel environments.')
    parser.add_argument('--task', type=str, default='pushing', help=f'Task name. Supported: {SUPPORTED_TASKS}')
    parser.add_argument('--log_dir', type=str, default='ppo_pushing_sb3', help='Log directory.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--skip_frame', type=int, default=3, help='Frame skip.')
    parser.add_argument('--max_episode_length', type=int, default=250, help='Max episode length.')
    parser.add_argument('--eval_freq', type=int, default=100_000, help='Evaluation frequency.')
    parser.add_argument('--use_wandb', action='store_true', help='Enable wandb logging.')
    args = parser.parse_args()

    if args.task not in TASK_BENCHMARKS:
        print(f"\nERROR: Task '{args.task}' is not supported for evaluation.")
        print(f"Supported tasks: {SUPPORTED_TASKS}")
        exit(1)

    # --- Tuned PPO config for CausalWorld tasks ---
    ppo_config = {
        'gamma': 0.995,
        'n_steps': 4096,  # longer rollouts for better credit assignment
        'ent_coef': 0.02, # encourage exploration
        'learning_rate': 2.5e-4, # slightly lower for stability
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'gae_lambda': 0.97,
        'batch_size': 512, # larger batch for more stable updates
        'n_epochs': 15
    }

    wandb_config = None
    if args.use_wandb:
        wandb_config = {
            'project': f'causal-world-ppo-{args.task}',
            'run_name': f"ppo_{args.task}_seed{args.seed}_{args.timesteps}steps"
        }

    if args.train:
        train_policy(
            num_envs=args.num_envs,
            log_dir=args.log_dir,
            max_episode_length=args.max_episode_length,
            skip_frame=args.skip_frame,
            seed=args.seed,
            ppo_config=ppo_config,
            total_timesteps=args.timesteps,
            eval_freq=args.eval_freq,
            task_name=args.task,
            wandb_config=wandb_config
        )
    if args.eval:
        eval(args.log_dir, task_name=args.task, seed=args.seed, max_episode_length=args.max_episode_length, skip_frame=args.skip_frame)
        gen_eval(args.log_dir, task_name=args.task, seed=args.seed)

def eval(log_dir, task_name='pushing', seed=0, max_episode_length=250, skip_frame=3, num_episodes=10):
    print("Running evaluation...")
    dense_weights = DENSE_REWARD_WEIGHTS.get(task_name, [0])
    task = generate_task(
        task_generator_id=task_name,
        dense_reward_weights=np.array(dense_weights),
        variables_space='space_a',
        fractional_reward_weight=1
    )
    env = CausalWorld(
        task=task,
        skip_frame=skip_frame,
        action_mode='joint_torques',
        enable_visualization=False,
        seed=seed,
        max_episode_length=max_episode_length
    )

    model_path = os.path.join(log_dir, 'final_model.zip')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}")

    model = PPO.load(model_path)

    all_rewards, all_successes = [], []
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward, successes = 0.0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if isinstance(info, dict) and 'success' in info:
                successes += int(info['success'])
        print(f"Episode {ep + 1}: Reward = {total_reward:.2f}, Success = {successes}")
        all_rewards.append(total_reward)
        all_successes.append(successes)

    print(f"\nMean Reward: {np.mean(all_rewards):.2f}")
    print(f"Mean Success Rate: {np.mean(all_successes):.2f}")

def gen_eval(log_dir, task_name='pushing', seed=0):
    print("Generating benchmark evaluation and visualization...")
    if task_name not in TASK_BENCHMARKS:
        print(f"No benchmark available for task '{task_name}'. Supported: {SUPPORTED_TASKS}")
        return
    benchmark = TASK_BENCHMARKS[task_name]
    # Use CausalWorld's benchmark
    model_path = os.path.join(log_dir, 'final_model.zip')
    evaluation = EvaluationPipeline(
        evaluation_protocols=benchmark['evaluation_protocols'],
        task_params={'task_generator_id': task_name},
        world_params={'skip_frame': 3, 'action_mode': 'joint_torques'},
        policy_class=PPO,
        policy_path=model_path,
        visualize_evaluation=False
    )
    def policy_fn(obs):
        model = PPO.load(model_path)
        action, _ = model.predict(obs, deterministic=True)
        return action
    scores_model = evaluation.evaluate_policy(policy_fn, fraction=0.005)
    print("\nEvaluation Results:")
    for metric, value in scores_model.items():
        if isinstance(value, (int, float)):
            print(f"• {metric.replace('_', ' ').title()}: {value:.2f}")
        else:
            print(f"• {metric.replace('_', ' ').title()}: {value}")
    vis.generate_visual_analysis(log_dir, experiments={task_name: scores_model})
    print("Visualization saved to:", log_dir)

if __name__ == '__main__':
    main()

# ---
# Would curriculum learning help?
# Yes! Curriculum learning (progressively increasing task difficulty) is well-known to improve final performance and sample efficiency in robotic manipulation tasks like CausalWorld. After pretraining a strong policy, you can continue training with a curriculum (e.g., more challenging object positions, more distractors, etc.) to further boost generalization and robustness.

# python ppo_vanilla.py --train --eval --use_wandb --timesteps 5000000 --num_envs 16 --task pushing --log_dir ppo_pushing_sb3
# python ppo_vanilla.py --train --eval --use_wandb --timesteps 5000000 --num_envs 16 --task picking --log_dir ppo_picking_sb3
# python ppo_vanilla.py --train --eval --use_wandb --timesteps 5000000 --num_envs 16 --task reaching --log_dir ppo_reaching_sb3
# python ppo_vanilla.py --train --eval --use_wandb --timesteps 5000000 --num_envs 16 --task pick_and_place --log_dir ppo_pick_and_place_sb3
# python ppo_vanilla.py --train --eval --use_wandb --timesteps 5000000 --num_envs 16 --task stacking2 --log_dir ppo_stacking2_sb3

# tasks:
# picking
# creative_stacked_blocks
# pick_and_place
# pushing
# reaching
# stacked_blocks
# stacking2
# towers

# benchmarks:
# REACHING_BENCHMARK
# PUSHING_BENCHMARK
# PICKING_BENCHMARK
# PICK_AND_PLACE_BENCHMARK
# STACKING2_BENCHMARK

# dense weights ->
# reaching = [100000, 0, 0, 0]
# pushing = [750, 250, 100]
# picking = [250, 0, 125, 0, 750, 0, 0, 0.005]
# pick_and_place = [750, 50, 250, 0, 0.005]
# stacking2 = [750, 250, 250, 125, 0.005]