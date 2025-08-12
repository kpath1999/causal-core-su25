import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Before other imports

from causal_world.task_generators.task import generate_task
from causal_world.envs.causalworld import CausalWorld
from stable_baselines3 import SAC
from stable_baselines3.sac import MultiInputPolicy  # Required for HER
from stable_baselines3.her import HerReplayBuffer
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
from causal_world.wrappers.env_wrappers import HERGoalEnvWrapper

import numpy as np
np.bool = bool  # Fix for numpy 1.20+ compatibility
import wandb
from wandb.integration.sb3 import WandbCallback

def train_policy(num_of_envs, log_relative_path, maximum_episode_length,
                 skip_frame, seed_num, her_config, total_time_steps,
                 validate_every_timesteps, task_name, wandb_config=None):
    
    # initialize the w&b run
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
                **her_config  # Log all hyperparameters
            },
            tags=[task_name, 'HER', 'SAC'],
            sync_tensorboard=True
        )
    
    # create the environment
    task = generate_task(task_generator_id=task_name,
                         dense_reward_weights=np.array([0]*8),
                         fractional_reward_weight=1,
                         goal_height=0.15,
                         tool_block_mass=0.02)

    env = CausalWorld(task=task,
                      skip_frame=skip_frame,
                      enable_visualization=False,
                      seed=seed_num,
                      max_episode_length=maximum_episode_length)

    # apply the HER wrapper
    final_env = HERGoalEnvWrapper(env)
    set_random_seed(seed_num)
    
    # time to callback
    callbacks = [
        CheckpointCallback(
            save_freq=int(validate_every_timesteps / num_of_envs),
            save_path=log_relative_path,
            name_prefix='model'
        )
    ]
    
    # Add W&B callback if configured
    if wandb_config:
        wandb_callback = WandbCallback(
            gradient_save_freq=10000,
            model_save_path=f"models/{wandb.run.id}",
            verbose=2,
        )
        callbacks.append(wandb_callback)

    # Extract HER-specific parameters
    replay_buffer_kwargs = {
        'n_sampled_goal': her_config.pop('n_sampled_goal', 4),
        'goal_selection_strategy': her_config.pop('goal_selection_strategy', 'future'),
        'max_episode_length': maximum_episode_length
    }

    model = SAC(
        MultiInputPolicy,  # REQUIRED for HER
        final_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=replay_buffer_kwargs,
        verbose=1,
        tensorboard_log=her_config.pop('tensorboard_log'),
        **her_config,
        seed=seed_num
    )
    
    model.learn(total_timesteps=total_time_steps,
                tb_log_name="her_sac",
                callback=callbacks)
    
    # finish w&b run
    if wandb_config:
        wandb.finish()
    
    return model

if __name__ == '__main__':

    print("This is a placeholder for the mfrl_scratch module.")
    a = np.zeros((10, 10))  # Example of a random array
    print("Random array generated:")
    print(a)

    # Training parameters
    total_time_steps_per_update = 1000000
    total_time_steps = 60000000
    num_of_envs = 20
    log_relative_path = 'baseline_picking_her_sac'
    maximum_episode_length = 600
    skip_frame = 3
    seed_num = 0
    task_name = 'picking'
    
    # HER and SAC parameters
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
        "tensorboard_log": log_relative_path
    }

    # W&B configuration
    wandb_config = {
        'project': 'causal-world-her-sac',
        'run_name': f'{task_name}_seed_{seed_num}_{maximum_episode_length}steps',
        'tags': ['her', 'sac']
    }

    model = train_policy(
        num_of_envs=num_of_envs,
        log_relative_path=log_relative_path,
        maximum_episode_length=maximum_episode_length,
        skip_frame=skip_frame,
        seed_num=seed_num,
        her_config=her_config,
        total_time_steps=total_time_steps,
        validate_every_timesteps=total_time_steps_per_update,
        task_name=task_name,
        wandb_config=wandb_config
    )