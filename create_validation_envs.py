"""
Script to create and save validation environments for each task.
This generates 10 different environment configurations for each task
and saves them as pickle files for later use in validation.
"""

import os
import pickle
import numpy as np
import random
import torch
import logging
from causal_world.task_generators import generate_task
from causal_world.envs import CausalWorld
from baselines import DENSE_REWARD_WEIGHTS, SUPPORTED_TASKS

def set_seed(seed):
    """set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_environment(task_name, seed=0, skip_frame=3, max_episode_length=500):
    """create a basic env without interventions"""
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

    # explicitly set seed for reproducibility
    if hasattr(env, 'seed'):
        env.seed(seed)
    
    return env

def create_and_save_validation_envs(base_seed=42, num_envs=10):
    """
    create and save val envs for each supported task

    args:
        base_seed: base seed to use for env creation
        num_envs: number of environments to create per task
    """
    # create base directory
    os.makedirs("envs", exist_ok=True)

    for task_name in SUPPORTED_TASKS:
        # create task directory
        task_dir = os.path.join("envs", task_name)
        os.makedirs(task_dir, exist_ok=True)

        logging.info(f"Creating {num_envs} validation environments for {task_name}")

        for env_idx in range(num_envs):
            # use different seed for each env
            seed = base_seed + env_idx * 100
            set_seed(seed)

            # create env
            env = create_environment(
                task_name=task_name,
                seed=seed,
                skip_frame=3,
                max_episode_length=500
            )

            # save the environment state
            env_filename = f"val_env_{env_idx}.pkl"
            env_path = os.path.join(task_dir, env_filename)

            # we pickle the the task parameters rather than the env itself
            task_params = {
                'task_name': task_name,
                'seed': seed,
                'variable_space': env.get_variable_space_used(),    # save current variable space
                'skip_frame': 3,
                'max_episode_length': 500
            }

            with open(env_path, 'wb') as f:
                pickle.dump(task_params, f)
            
            logging.info(f"    Saved {task_name} validation environment {env_idx} to {env_path}")

            # close the environment
            env.close()
        
        logging.info(f"Completed {task_name} validation environments")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )
    create_and_save_validation_envs()
    logging.info("All validation environments created successfully!")