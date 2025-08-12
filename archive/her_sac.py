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
import torch

from curriculum.graph_interventions import GraphBasedCurriculumManager, GraphBasedCurriculumCallback
from causal_learning import (
    CausalGNN, CausalGraph, CausalGraphNode,
    CausalLearningDashboard, CausalInterpretability,
    CausalLearningMonitor
)

class CausalAwareHER:
    def __init__(self, causal_gnn, policy):
        self.causal_gnn = causal_gnn
        self.policy = policy
        self.causal_graph = CausalGraph()
        self.interpreter = CausalInterpretability()
        self.monitor = CausalLearningMonitor()
        self.dashboard = CausalLearningDashboard()
        
    def update_causal_graph(self, observation, action, next_observation):
        # Update node states
        node_states = self.extract_node_states(observation)
        
        # Update edge relationships
        edge_index = self.compute_edge_index(node_states)
        
        # Predict causal effects
        causal_effects, attention_weights = self.causal_gnn(node_states, edge_index)
        
        # Update causal graph
        self.update_graph_structure(causal_effects)
        
        # Update interpretability
        self.interpreter.attention_weights = attention_weights
        
        # Update monitoring
        self.monitor.update_metrics({
            'causal_effects': causal_effects.detach().numpy(),
            'attention_weights': attention_weights.detach().numpy()
        })
        
    def select_intervention(self, state):
        # Use causal graph to select informative interventions
        node_importance = self.compute_node_importance()
        intervention_target = self.select_target_node(node_importance)
        return self.generate_intervention(intervention_target)

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
                'graph_curriculum_enabled': True,
                **her_config  # Log all hyperparameters
            },
            tags=[task_name, 'HER', 'SAC', 'GraphCurriculum', 'CausalLearning'],
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

    # apply the HER wrapper
    final_env = HERGoalEnvWrapper(curriculum_env)
    set_random_seed(seed_num)
    
    # Initialize causal learning components
    causal_gnn = CausalGNN(node_dim=64, edge_dim=32)
    causal_her = CausalAwareHER(causal_gnn, None)
    
    # time to callback
    callbacks = [
        CheckpointCallback(
            save_freq=int(validate_every_timesteps / num_of_envs),
            save_path=log_relative_path,
            name_prefix='model'
        )
    ]
    # add graph-based curriculum callback
    graph_curriculum_cb = GraphBasedCurriculumCallback(
        curriculum_manager=graph_curriculum_manager,
        curriculum_wrapper_env=curriculum_env,
        adaptation_interval_episodes=30
    )
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
    
    # Set the policy in causal HER
    causal_her.policy = model
    
    model.learn(total_timesteps=total_time_steps,
                tb_log_name="her_sac_curriculum",
                callback=callbacks)
    
    # finish w&b run
    if wandb_config:
        wandb.finish()
    
    return model, causal_her

if __name__ == '__main__':
    # Enhanced training parameters
    total_time_steps_per_update = 1000000
    total_time_steps = 2000000     # Increased from 50k to 2M steps for thorough curriculum learning
    num_of_envs = 20
    log_relative_path = 'baseline_picking_her_sac_curriculum'
    maximum_episode_length = 400    # Increased from 200 to 400 for more complex behaviors
    skip_frame = 3
    seed_num = 0
    task_name = 'picking'
    
    # Enhanced HER and SAC parameters
    her_config = {
        "n_sampled_goal": 4,  # Increased from 2 to 4 for better goal sampling
        "goal_selection_strategy": 'future',
        "gamma": 0.99,        # Increased from 0.98 for better long-term planning
        "tau": 0.005,         # Reduced from 0.01 for more stable learning
        "ent_coef": 'auto',
        "target_entropy": -12,  # Adjusted for longer episodes
        "learning_rate": 0.0003,  # Slightly increased for faster learning
        "buffer_size": 1000000,  # Increased from 500k to 1M for better experience replay
        "learning_starts": 5000,  # Increased from 1000 for better initial exploration
        "batch_size": 256,     # Increased from 128 for better gradient estimates
        "tensorboard_log": log_relative_path
    }

    # Enhanced W&B configuration
    wandb_config = {
        'project': 'causal-world-her-sac-graph-curriculum',
        'run_name': f'{task_name}_seed_{seed_num}_{maximum_episode_length}steps_curriculum',
        'tags': ['curriculum_learning', 'graph_based', 'causal_impact', 'causal_learning']
    }

    model, causal_her = train_policy(
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
    
    # Create and run the dashboard
    dashboard = causal_her.dashboard.create_dashboard(causal_her)
    dashboard.run_server(debug=True, port=8050)