import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList, BaseCallback
import wandb
from wandb.integration.sb3 import WandbCallback

from causal_world.task_generators import generate_task
from causal_world.envs import CausalWorld
from causal_world.wrappers.curriculum_wrappers import CurriculumWrapper
from causal_world.intervention_actors import GoalInterventionActorPolicy, PhysicalPropertiesInterventionActorPolicy

# --- Curriculum definition ---
curriculum_kwargs = {
    'intervention_actors': [
        GoalInterventionActorPolicy(),  # Vary position/goal
        PhysicalPropertiesInterventionActorPolicy(group='mass'),  # Vary mass
        PhysicalPropertiesInterventionActorPolicy(group='friction'),  # Vary friction
    ],
    'actives': [
        (0, 100000, 1, 0),   # Goal intervention: episodes 0-100k, every episode, timestep 0
        (100000, 200000, 1, 0),  # Mass intervention: episodes 100k-200k, every episode, timestep 0
        (200000, 300000, 1, 0),  # Friction intervention: episodes 200k-300k, every episode, timestep 0
    ]
}

DENSE_REWARD_WEIGHTS = {
    'pushing': [750, 250, 100],
}

def make_env_with_curriculum(rank, task_name, seed=0, skip_frame=3, max_episode_length=250, log_dir=None):
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
        env = CurriculumWrapper(env, **curriculum_kwargs)
        return env
    return _init

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

def train_with_curriculum(num_envs, log_dir, max_episode_length, skip_frame, seed, ppo_config, total_timesteps, eval_freq, task_name, pretrained_path, wandb_config=None):
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
            tags=[task_name, 'PPO', 'SB3', 'curriculum'],
            sync_tensorboard=True
        )

    os.makedirs(log_dir, exist_ok=True)
    env = SubprocVecEnv([
        make_env_with_curriculum(i, task_name, seed, skip_frame, max_episode_length, log_dir=log_dir)
        for i in range(num_envs)
    ])
    env = VecMonitor(env, filename=os.path.join(log_dir, 'monitor.csv'))

    eval_env = make_env_with_curriculum(0, task_name, seed, skip_frame, max_episode_length, log_dir=log_dir)()

    policy_kwargs = dict(activation_fn=nn.LeakyReLU, net_arch=[512, 256])
    model = PPO.load(pretrained_path, env=env, policy_kwargs=policy_kwargs, tensorboard_log=log_dir, seed=seed, **ppo_config)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(eval_freq // num_envs, 1),
        save_path=os.path.join(log_dir, 'logs'),
        name_prefix='ppo_model_curr'
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
        tb_log_name="ppo_sb3_curr_basic"
    )
    model.save(os.path.join(log_dir, 'final_model_curr'))
    if wandb_config:
        wandb.finish()
    print(f"Final curriculum model saved to {os.path.join(log_dir, 'final_model_curr')}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model with curriculum.')
    parser.add_argument('--timesteps', type=int, default=1_000_000, help='Total timesteps.')
    parser.add_argument('--num_envs', type=int, default=16, help='Number of parallel environments.')
    parser.add_argument('--log_dir', type=str, default='ppo_pushing_sb3_curr', help='Log directory.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--skip_frame', type=int, default=3, help='Frame skip.')
    parser.add_argument('--max_episode_length', type=int, default=250, help='Max episode length.')
    parser.add_argument('--eval_freq', type=int, default=100_000, help='Evaluation frequency.')
    parser.add_argument('--use_wandb', action='store_true', help='Enable wandb logging.')
    parser.add_argument('--pretrained_path', type=str, default='ppo_pushing_sb3/final_model.zip', help='Path to pretrained PPO model.')
    args = parser.parse_args()

    ppo_config = {
        'n_steps': 2048 // args.num_envs,
        'batch_size': 256,
        'learning_rate': 3e-4,
        'gamma': 0.98,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
    }

    wandb_config = None
    if args.use_wandb:
        wandb_config = {
            'project': 'causal-world-ppo-curriculum',
            'run_name': f'pushing_curr_seed_{args.seed}_{args.max_episode_length}steps'
        }

    if args.train:
        train_with_curriculum(
            num_envs=args.num_envs,
            log_dir=args.log_dir,
            max_episode_length=args.max_episode_length,
            skip_frame=args.skip_frame,
            seed=args.seed,
            ppo_config=ppo_config,
            total_timesteps=args.timesteps,
            eval_freq=args.eval_freq,
            task_name='pushing',
            pretrained_path=args.pretrained_path,
            wandb_config=wandb_config
        )

if __name__ == '__main__':
    main() 

# python ppo_pushing_curr_basic.py --train --timesteps 1000000 --num_envs 16 --use_wandb --pretrained_path ppo_pushing_sb3/final_model.zip