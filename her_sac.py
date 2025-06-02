import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Before other imports

from causal_world.task_generators.task import generate_task
from causal_world.envs.causalworld import CausalWorld
from stable_baselines3 import HER, SAC
from stable_baselines3.sac.policies import MlpPolicy

from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
from causal_world.wrappers.env_wrappers import HERGoalEnvWrapper
from causal_world.intervention_actors import GoalInterventionActorPolicy, RandomInterventionActorPolicy, VisualInterventionActorPolicy

# refer to: https://github.com/rr-learning/CausalWorld/blob/master/causal_world/wrappers/curriculum_wrappers.py
from causal_world.wrappers.curriculum_wrappers import CurriculumWrapper

from causal_world.task_generators.task import generate_task
from causal_world.envs.causalworld import CausalWorld
from stable_baselines3 import SAC
from stable_baselines3.sac import MultiInputPolicy  # Required for HER
from stable_baselines3.her import HerReplayBuffer

import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback

from curriculum.graph_interventions import GraphBasedCurriculumManager, GraphBasedCurriculumCallback

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
                'curriculum_enabled': curriculum_kwargs is not None,
                **her_config  # Log all hyperparameters
            },
            tags=[task_name, 'HER', 'SAC'],
            sync_tensorboard=True  # Also sync tensorboard logs
        )
    
    # create the environment FIRST
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

    # initialize the graph-based curriculum manager
    graph_curriculum_manager = GraphBasedCurriculumManager(total_timesteps=total_time_steps)

    # get initial curriculum configuration
    initial_curriculum_config = graph_curriculum_manager.get_current_curriculum_config()

    # apply curriculum wrapper with graph-based curriculum
    curriculum_env = CurriculumWrapper(
        env,
        intervention_actors=initial_curriculum_config['intervention_actors'],
        actives=initial_curriculum_config['actives']
    )

    final_env = HERGoalEnvWrapper(curriculum_env)
    set_random_seed(seed_num)
    
    # add graph-based curriculum callback
    graph_curriculum_cb = GraphBasedCurriculumCallback(
        curriculum_manager=graph_curriculum_manager,
        curriculum_wrapper_env=curriculum_env,
        adaptation_interval_episodes=30
    )
    # combining callbacks
    callbacks = [
        CheckpointCallback(
            save_freq=int(validate_every_timesteps / num_of_envs),
            save_path=log_relative_path,
            name_prefix='model'
        )
    ]
    callbacks.append(graph_curriculum_cb)
    
    # Add W&B callback if configured
    if wandb_config:
        wandb_callback = WandbCallback(
            gradient_save_freq=10000,  # Save gradients every 10k steps
            model_save_path=f"models/{wandb.run.id}",
            verbose=2,
        )
        callbacks.append(wandb_callback)

    # Extract HER-specific parameters
    replay_buffer_kwargs = {
        'n_sampled_goal': her_config.pop('n_sampled_goal', 2),  # reduced from 4
        'goal_selection_strategy': her_config.pop('goal_selection_strategy', 'future'),
        'max_episode_length': maximum_episode_length
    }

    model = SAC(
        MultiInputPolicy,  # REQUIRED for HER
        final_env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=replay_buffer_kwargs,
        verbose=1,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=her_config.pop('tensorboard_log'),
        **her_config,
        seed=seed_num
    )
    
    model.learn(total_timesteps=total_time_steps,
                tb_log_name="her_sac_curriculum" if curriculum_kwargs else "her_sac",
                callback=callbacks)
    
    # finish w&b run
    if wandb_config:
        wandb.finish()
    
    return

if __name__ == '__main__':
    total_time_steps_per_update = 1000000
    total_time_steps = 50000     # this takes a while; 50k steps takes 45 mins; was 60M
    num_of_envs = 20
    log_relative_path = 'baseline_picking_her_sac_curriculum'
    maximum_episode_length = 200    # reduced from 600
    skip_frame = 3
    seed_num = 0
    task_name = 'picking'
    her_config = {
        "n_sampled_goal": 2,  # Reduced from 4
        "goal_selection_strategy": 'future',
        "gamma": 0.98,
        "tau": 0.01,
        "ent_coef": 'auto',
        "target_entropy": -9,
        "learning_rate": 0.00025,
        "buffer_size": 500000,  # Reduced from 1,000,000
        "learning_starts": 1000,
        "batch_size": 128,  # Reduced from 256
        "tensorboard_log": log_relative_path
    }

    # w&b configuration
    wandb_config = {
        'project': 'causal-world-her-sac-curriculum',
        'run_name': f'{task_name}_seed_{seed_num}_{maximum_episode_length}steps'
    }

    train_policy(
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