"""
===== TERMINAL COMMANDS FOR BASELINE TRAINING =====

This section provides all possible terminal commands to generate logs from the baselines.
Copy and paste any of these commands to run different curriculum learning experiments.

BASIC TRAINING COMMANDS:
-----------------------

1. GREEDY CURRICULUM (highest reward intervention selection):
python baselines.py --train --curriculum_mode greedy --task pushing --timesteps 50000 --use_wandb

2. CAUSAL MISMATCH CURRICULUM (CM score-based selection):
python baselines.py --train --curriculum_mode cm --task pushing --timesteps 50000 --alpha_cm 0.5 --use_wandb

3. RANDOM CURRICULUM (random intervention selection):
python baselines.py --train --curriculum_mode random --task pushing --timesteps 50000 --use_wandb

4. NO CURRICULUM (baseline without interventions):
python baselines.py --train --curriculum_mode none --task pushing --timesteps 50000 --use_wandb

5. RND INTRINSIC MOTIVATION (Random Network Distillation):
python baselines.py --train --curriculum_mode rnd --task pushing --timesteps 50000 --rnd_beta 0.01 --rnd_update_freq 1000 --rnd_batch_size 1024 --use_wandb

6. COUNT-BASED EXPLORATION (state visitation counts):
python baselines.py --train --curriculum_mode count --task pushing --timesteps 50000 --count_beta 0.01 --count_encoding_dim 32 --use_wandb

7. LEARNING PROGRESS MOTIVATION (improvement in transition model prediction accuracy):
python baselines.py --train --curriculum_mode lpm --task pushing --timesteps 50000 --lpm_beta 1.0 --use_wandb

8. INFORMATION GAIN REWARD (rewards "surprise" measured by model uncertainty):
python baselines.py --train --curriculum_mode info --task pushing --timesteps 50000 --info_beta 1.0 --use_wandb

REPLACEMENT MODE (allow same intervention to be selected multiple times):
------------------------------------------------------------------------

9. GREEDY CURRICULUM WITH REPLACEMENT:
python baselines.py --train --curriculum_mode greedy --task pushing --timesteps 50000 --replacement --use_wandb

10. CAUSAL MISMATCH CURRICULUM WITH REPLACEMENT:
python baselines.py --train --curriculum_mode cm --task pushing --timesteps 50000 --alpha_cm 0.5 --replacement --use_wandb

11. RANDOM CURRICULUM WITH REPLACEMENT:
python baselines.py --train --curriculum_mode random --task pushing --timesteps 50000 --replacement --use_wandb

LOG DIRECTORY STRUCTURE:
-----------------------

All logs are automatically organized in a centralized 'logs/' directory:

- logs/greedy_sequencing_logs/              (greedy without replacement)
- logs/greedy_replacement_sequencing_logs/  (greedy with replacement)
- logs/cm_sequencing_logs/                  (causal mismatch without replacement)
- logs/cm_replacement_sequencing_logs/      (causal mismatch with replacement)
- logs/random_sequencing_logs/              (random without replacement)
- logs/random_replacement_sequencing_logs/  (random with replacement)
- logs/none_sequencing_logs/                (no curriculum baseline)
- logs/rnd_sequencing_logs/                 (RND intrinsic motivation)
- logs/count_sequencing_logs/               (count-based exploration)

CUSTOM LOG DIRECTORY:
--------------------

12. SPECIFY CUSTOM LOG DIRECTORY:
python baselines.py --train --curriculum_mode greedy --task pushing --timesteps 50000 --log_dir custom_experiment --use_wandb

MODEL DIRECTORY STRUCTURE:
-------------------------

All pretrained models are expected to be in a centralized 'models/' directory:

- models/ppo_pushing_sb3/final_model.zip
- models/ppo_reaching_sb3/final_model.zip  
- models/ppo_picking_sb3/final_model.zip
- models/ppo_pick_and_place_sb3/final_model.zip
- models/ppo_stacking2_sb3/final_model.zip

13. SPECIFY CUSTOM PRETRAINED MODEL PATH:
python baselines.py --train --curriculum_mode greedy --task pushing --timesteps 50000 --pretrained_path path/to/custom_model.zip --use_wandb

Note: Replace 'pushing' with any supported task: reaching, picking, pick_and_place, stacking2
Note: All commands assume you have the required pretrained models in models/{task} directories (e.g., models/ppo_pushing_sb3/final_model.zip)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import argparse
import logging
import time
import csv
from collections import deque
from copy import deepcopy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
from causal_world.task_generators import generate_task
from causal_world.envs import CausalWorld
from causal_world.evaluation import EvaluationPipeline
from causal_world.benchmark import REACHING_BENCHMARK, PUSHING_BENCHMARK, PICKING_BENCHMARK, PICK_AND_PLACE_BENCHMARK, STACKING2_BENCHMARK
import causal_world.evaluation.visualization.visualiser as vis
from causal_world.intervention_actors import (
    GoalInterventionActorPolicy, 
    PhysicalPropertiesInterventionActorPolicy,
    VisualInterventionActorPolicy,
    RigidPoseInterventionActorPolicy,
    RandomInterventionActorPolicy
)
from validation_actor import ValidationInterventionActorPolicy
import wandb
from wandb.integration.sb3 import WandbCallback

# =====================
# Utility: Set random seed
# =====================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logging.info(f"Random seed set to {seed}")

# =====================
# Interventions List
# =====================
INTERVENTIONS = [
    {"type": "goal", "class": GoalInterventionActorPolicy, "params": {}},
    {"type": "mass", "class": PhysicalPropertiesInterventionActorPolicy, "params": {"group": "tool"}},
    {"type": "friction", "class": PhysicalPropertiesInterventionActorPolicy, "params": {"group": "stage"}},
    {"type": "visual", "class": VisualInterventionActorPolicy, "params": {}},
    {"type": "position", "class": RigidPoseInterventionActorPolicy, "params": {"positions": True, "orientations": False}},
    {"type": "angle", "class": RigidPoseInterventionActorPolicy, "params": {"positions": False, "orientations": True}},
    {"type": "random", "class": RandomInterventionActorPolicy, "params": {}}
]
TASK_BENCHMARKS = {
    'reaching': REACHING_BENCHMARK,
    'pushing': PUSHING_BENCHMARK,
    'picking': PICKING_BENCHMARK,
    'pick_and_place': PICK_AND_PLACE_BENCHMARK,
    'stacking2': STACKING2_BENCHMARK
}
SUPPORTED_TASKS = list(TASK_BENCHMARKS.keys())

# =====================
# DENSE_REWARD_WEIGHTS
# =====================
DENSE_REWARD_WEIGHTS = {
    'pushing': [750, 250, 100],
    'picking': [250, 0, 125, 0, 750, 0, 0, 0.005],
    'reaching': [100000, 0, 0, 0],
    'pick_and_place': [750, 50, 250, 0, 0.005],
    'stacking2': [750, 250, 250, 125, 0.005],
}

# =====================
# RND Intrinsic Reward Model
# =====================
class RNDIntrinsicModel:
    """RND model for computing intrinsic rewards based on prediction errors"""
    def __init__(self, input_dim, device='cpu', hidden_dim=256, output_dim=128):
        self.device = device

        # target network (frozen)
        self.target = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ).to(device)

        # predictor network (trainable)
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ).to(device)

        # freeze target network parameters
        for param in self.target.parameters():
            param.requires_grad = False
        
        # initialize predictor optimizer
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=1e-3)

        # normalization parameters
        self.obs_mean = None
        self.obs_std = None
    
    def normalize_obs(self, obs):
        """normalize observations for stable training"""
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)

        if self.obs_mean is None:
            self.obs_mean = obs_tensor.mean(dim=0, keepdim=True)
            self.obs_std = obs_tensor.std(dim=0, keepdim=True) + 1e-8
        
        # update running statistics
        momentum = 0.99
        self.obs_mean = momentum * self.obs_mean + (1 - momentum) * obs_tensor.mean(dim=0, keepdim=True)
        self.obs_std = momentum * self.obs_std + (1 - momentum) * obs_tensor.std(dim=0, keepdim=True) + 1e-8

        normalized = (obs_tensor - self.obs_mean) / self.obs_std
        return torch.clamp(normalized, -5, 5)

    def compute_intrinsic_reward(self, obs_batch):
        """compute the intrinsic reward as prediction error"""
        normalized_obs = self.normalize_obs(obs_batch)

        with torch.no_grad():
            target_features = self.target(normalized_obs)
        
        predicted_features = self.predictor(normalized_obs)
        prediction_errors = (predicted_features - target_features).pow(2).mean(dim=1)

        return prediction_errors.detach().cpu().numpy()
    
    def train_predictor(self, obs_batch):
        """train the predictor network"""
        normalized_obs = self.normalize_obs(obs_batch)

        with torch.no_grad():
            target_features = self.target(normalized_obs)
        
        predicted_features = self.predictor(normalized_obs)
        loss = (predicted_features - target_features).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

# =====================
# RND Reward Callback
# =====================
class RNDIntrinsicRewardCallback(BaseCallback):
    """callback that augments env rewards with RND intrinsic rewards"""
    def __init__(self, beta=0.01, update_freq=1000, batch_size=1024, device='cpu', verbose=0):
        super(RNDIntrinsicRewardCallback, self).__init__(verbose)
        self.beta = beta                    # intrinsic reward scaling factor
        self.update_freq = update_freq      # how often to train the predictor
        self.batch_size = batch_size        # training batch size
        self.device = device
        self.rnd_model = None
        self.obs_buffer = []
        self.last_update_step = 0

        # reward normalization
        self.reward_rms = None
        self.intrinsic_reward_history = []

    def _on_training_start(self):
        """initialize RND model when training starts"""
        obs_space = self.training_env.observation_space
        input_dim = int(np.prod(obs_space.shape))

        self.rnd_model = RNDIntrinsicModel(input_dim, self.device)

        if self.verbose > 0:
            print(f"[RND] initialized with input dimension: {input_dim}")
            print(f"[RND] beta (intrinsic reward scale): {self.beta}")
            print(f"[RND] update frequency: {self.update_freq}")
    
    def _on_step(self):
        """called after each env step"""
        # access observations and rewards from PPO
        obs = self.locals.get('obs')
        rewards = self.locals.get('rewards')

        if obs is None or rewards is None:
            return True

        # flatten observations if needed
        obs_processed = obs.copy()
        if len(obs_processed.shape) > 2:
            obs_processed = obs_processed.reshape(obs_processed.shape[0], -1)
        
        # compute intrinsic rewards
        intrinsic_rewards = self.rnd_model.compute_intrinsic_reward(obs_processed)

        # normalize intrinsic rewards using RMS
        self.intrinsic_reward_history.extend(intrinsic_rewards)
        if len(self.intrinsic_reward_history) > 10000:     # keep the last 10k rewards
            self.intrinsic_reward_history = self.intrinsic_reward_history[-10000:]
        
        if len(self.intrinsic_reward_history) > 100:
            intrinsic_std = np.std(self.intrinsic_reward_history)
            if intrinsic_std > 0:
                intrinsic_rewards = intrinsic_rewards / (intrinsic_std + 1e-8)
        
        # augment env rewards with intrinsic rewards
        augmented_rewards = rewards + self.beta * intrinsic_rewards

        # update rewards in-place (this affects PPO's learning)
        self.locals['rewards'] = augmented_rewards

        # store observations for predictor training
        self.obs_buffer.extend(obs_processed)

        # periodically train the predictor network
        if (self.num_timesteps - self.last_update_step >= self.update_freq and len(self.obs_buffer) >= self.batch_size):
            # sample a batch for training
            batch_indices = np.random.choice(len(self.obs_buffer), self.batch_size, replace=False)
            batch = np.array([self.obs_buffer[i] for i in batch_indices])

            # train predictor
            loss = self.rnd_model.train_predictor(batch)

            if self.verbose > 0:
                avg_intrinsic = np.mean(intrinsic_rewards)
                print(f"[RND] step {self.num_timesteps}: predictor loss={loss:.6f}, avg intrinsic loss={avg_intrinsic:.4f}")
            
            # log to wandb if available
            if wandb.run is not None:
                wandb.log({
                    'rnd/predictor_loss': loss,
                    'rnd/avg_intrinsic_reward': np.mean(intrinsic_rewards),
                    'rnd/intrinsic_reward_std': np.std(intrinsic_rewards),
                    'step': self.num_timesteps
                })
            
            # clean up the buffer (keep only recent observations)
            self.obs_buffer = [self.obs_buffer[i] for i in range(len(self.obs_buffer)) if i not in batch_indices]
            self.last_update_step = self.num_timesteps
        
        return True

# =====================
# CM Score Models
# =====================
class TransitionPrediction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, state_dim)
    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
    
class RewardPrediction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class BetaVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=16, beta=4.0):
        super().__init__()
        # added a hidden layer for better capacity
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
        self.beta = beta
        self._init_weights()
    def _init_weights(self):
        """initialize weights to prevent initial instability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    def reparameterize(self, mu, logvar):
        # get a generator with a seed from the current pytorch global seed
        generator = torch.Generator(device=mu.device)
        generator.manual_seed(int(torch.randint(high=2**32-1, size=(1,)).item()))
        # clamping logvar before exponentiation
        logvar = torch.clamp(logvar, min=-10, max=2)   # conservative upper bound
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std, generator=generator)
        return mu + eps * std
    def forward(self, x):
        # normalize the input if not done already
        x = torch.clamp(x, min=-10, max=10)
        mu_logvar = self.encoder(x)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar


# =====================
# Evaluate CM Score
# =====================
def evaluate_cm_score(env, student_model, max_episodes=10, max_episode_length=500, device='cpu', intervention_type="unknown", seed=0):
    set_seed(seed)
    print(f"evaluating CM score for {intervention_type} intervention...")
    # collect data using student policy
    data = []
    total_steps = 0
    total_reward = 0
    success_count = 0
    termination_reasons = []
    for episode in range(max_episodes):
        # adding a unique seed per episode
        obs = env.reset(seed=seed+episode)
        done = False
        episode_steps = 0
        episode_reward = 0
        while not done:
            # using deterministic=True for evaluation
            act, _ = student_model.predict(obs, deterministic=True)
            next_obs, rew, done, info = env.step(act)
            # log the termination reason
            if done:
                if isinstance(info, dict) and 'success' in info and info['success']:
                    termination_reasons.append('success')
                    success_count += 1
                elif episode_steps >= max_episode_length:
                    termination_reasons.append('max_length')
                else:
                    termination_reasons.append('other')
            data.append((obs, act, next_obs, rew))
            obs = next_obs
            episode_steps += 1
            episode_reward += rew
            total_steps += 1
        total_reward += episode_reward
        if episode < 3:
            print(f"episode {episode+1}: {episode_steps} steps, reward: {episode_reward:.3f}")
        print(f"total data points collected: {len(data)}")
        print(f"average episode length: {total_steps/max_episodes:.1f}")
        print(f"average episode reward: {total_reward/max_episodes:.3f}")
        print(f"termination reasons: {termination_reasons}")
        print(f"success rate: {success_count}/{max_episodes}")
        if len(data) == 0:
            print("no data collected! returning cm score of 0")

    states = torch.tensor([d[0] for d in data], dtype=torch.float32).to(device)
    actions = torch.tensor([d[1] for d in data], dtype=torch.float32).to(device)    
    next_states = torch.tensor([d[2] for d in data], dtype=torch.float32).to(device)
    rewards = torch.tensor([d[3] for d in data], dtype=torch.float32).to(device).unsqueeze(-1)

    print(f"tensor shapes - states: {states.shape}, actions: {actions.shape}")

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    hidden_dim = 64

    # create ensembles
    print(f"Creating ensembles (obs_dim={obs_dim}, act_dim={act_dim}, hidden_dim={hidden_dim})")
    transition_models = [TransitionPrediction(obs_dim, act_dim, hidden_dim).to(device) for _ in range(5)]
    reward_models = [RewardPrediction(obs_dim, act_dim, hidden_dim).to(device) for _ in range(5)]
    state_models = [BetaVAE(obs_dim).to(device) for _ in range(5)]
    action_models = [BetaVAE(act_dim).to(device) for _ in range(5)]

    # create optimizers
    # NOTE: lowered vae learning rate
    transition_opts = [optim.Adam(m.parameters(), lr=1e-3) for m in transition_models]
    reward_opts = [optim.Adam(m.parameters(), lr=1e-3) for m in reward_models]
    state_opts = [optim.Adam(m.parameters(), lr=1e-4) for m in state_models]
    action_opts = [optim.Adam(m.parameters(), lr=1e-4) for m in action_models]

    # train transition models
    print("Training transition models...")
    transition_losses = []
    for model, opt in zip(transition_models, transition_opts):
        model_losses = []
        for iteration in range(5):
            pred = model(states, actions)
            loss = nn.MSELoss()(pred, next_states)
            opt.zero_grad()
            loss.backward()
            opt.step()
            model_losses.append(loss.item())
        transition_losses.append(np.mean(model_losses))
    print(f"Transition model losses: {[f'{l:.4f}' for l in transition_losses]}")
    
    # train reward models
    print(f"Training reward models...")
    reward_losses = []
    for model, opt in zip(reward_models, reward_opts):
        model_losses = []
        for iteration in range(5):
            pred = model(states, actions)
            loss = nn.MSELoss()(pred, rewards)
            opt.zero_grad()
            loss.backward()
            opt.step()
            model_losses.append(loss.item())
        reward_losses.append(np.mean(model_losses))
    print(f"Reward model losses: {[f'{l:.4f}' for l in reward_losses]}")

    # NOTE: i bumped up the training iterations for the vae models
    # train state VAE models
    print(f"Training state VAE models...")

    def preprocess_states(states):
        """normalize states to prevent vae training issues"""
        # remove any nan or inf values
        states = torch.nan_to_num(states, nan=0.0, posinf=10.0, neginf=-10.0)

        # standardize states (zero mean, unit variance)
        mean = states.mean(dim=0, keepdim=True)
        std = states.std(dim=0, keepdim=True) + 1e-8
        states_normalized = (states - mean) / std

        # clip to reasonable range
        states_normalized = torch.clamp(states_normalized, min=-3, max=3)
        return states_normalized

    # apply preprocessing
    states_processed = preprocess_states(states)

    # train state VAE models
    print(f"Training state VAE models...")
    state_losses = []

    for model, opt in zip(state_models, state_opts):
        model_losses = []
        consecutive_nans = 0

        for iteration in range(25):
            try:
                recon, mu, logvar = model(states_processed)
                # more stable kl computation
                kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                recon_loss = nn.MSELoss()(recon, states_processed)
                # check for nan before combining losses
                if torch.isnan(recon_loss) or torch.isnan(kl_div):
                    consecutive_nans += 1
                    if consecutive_nans > 5:    # too many consecutive nans
                        print(f"too many nan values - reinitializing model")
                        model._init_weights()
                        consecutive_nans = 0
                    continue
                
                # reduce beta for more stable training
                loss = recon_loss + (model.beta * 0.1) * kl_div

                if torch.isnan(loss):
                    consecutive_nans += 1
                    continue
                
                # reset the consecutive nan counter upon a successful iteration
                consecutive_nans = 0

                opt.zero_grad()
                loss.backward()

                # aggressive gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

                opt.step()
                model_losses.append(loss.item())
            
            except Exception as e:
                print(f"error in state vae training: {e}")
                continue
                
        # handle case where no valid losses were computed
        if model_losses:
            avg_loss = np.mean(model_losses)
        else:
            print(f"no valid losses computed - using fallback value")
            avg_loss = 1.0  # fallback value
        
        state_losses.append(avg_loss)
    
    print(f"State VAE losses: {[f'{l:.4f}' for l in state_losses]}")

    # train action VAE models
    print(f"Training action VAE models...")
    action_losses = []

    # preprocess actions to ensure stability
    def preprocess_actions(actions):
        """normalize actions to prevent vae training issues"""
        # remove any nan or inf values
        actions = torch.nan_to_num(actions, nan=0.0, posinf=10.0, neginf=-10.0)

        # standardize actions (zero mean, unit variance)
        mean = actions.mean(dim=0, keepdim=True)
        std = actions.std(dim=0, keepdim=True) + 1e-8
        actions_normalized = (actions - mean) / std

        # clip to reasonable range
        actions_normalized = torch.clamp(actions_normalized, min=-3, max=3)
        return actions_normalized

    # apply preprocessing
    actions_processed = preprocess_actions(actions)

    # add gradient clipping and loss checking in the action VAE training loop
    for model, opt in zip(action_models, action_opts):
        model_losses = []
        consecutive_nans = 0

        for iteration in range(25):
            try:
                recon, mu, logvar = model(actions_processed)
                # more stable kl computation
                kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                recon_loss = nn.MSELoss()(recon, actions_processed)
                # check for nan before combining losses
                if torch.isnan(recon_loss) or torch.isnan(kl_div):
                    consecutive_nans += 1
                    if consecutive_nans > 5:    # too many consecutive nans
                        print(f"too many nan values - reinitializing model")
                        model._init_weights()
                        consecutive_nans = 0
                    continue

                # reduce beta for more stable training
                loss = recon_loss + (model.beta * 0.1) * kl_div

                if torch.isnan(loss):
                    consecutive_nans += 1
                    continue
                
                # reset the consecutive nan counter upon a successful iteration
                consecutive_nans = 0

                opt.zero_grad()
                loss.backward()

                # aggressive gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

                opt.step()
                model_losses.append(loss.item())
            
            except Exception as e:
                print(f"error in action vae training: {e}")
                continue
        
        # handle case where no valid losses were computed
        if model_losses:
            avg_loss = np.mean(model_losses)
        else:
            print(f"no valid losses computed - using fallback value")
            avg_loss = 1.0  # fallback value
        
        action_losses.append(avg_loss)
    
    print(f"Action VAE losses: {[f'{l:.4f}' for l in action_losses]}")
    
    # calculate disagreement scores
    t_score = torch.stack([m(states, actions) for m in transition_models]).std(dim=0).mean().item()
    r_score = torch.stack([m(states, actions) for m in reward_models]).std(dim=0).mean().item()
    s_score = torch.stack([m(states_processed)[0] for m in state_models]).std(dim=0).mean().item()
    a_score = torch.stack([m(actions_processed)[0] for m in action_models]).std(dim=0).mean().item()
    cm_score = t_score + r_score + s_score + a_score

    print(f"CM score components:")
    print(f"transition disagreement: {t_score:.4f}")
    print(f"reward disagreement: {r_score:.4f}")
    print(f"state disagreement: {s_score:.4f}")
    print(f"action disagreement: {a_score:.4f}")
    print(f"total CM score: {cm_score:.4f}")

    return cm_score

# =====================
# Learning Progress Motivation (LPM) Model
# =====================
class LPMRewardCallback(BaseCallback):
    """learning progress motivation (LPM): reward based on improvement in model prediction"""
    def __init__(self, beta=1.0, lr=1e-3, batch_size=256, n_train_steps=1, buffer_size=50000, device='cpu', verbose=0):
        super(LPMRewardCallback, self).__init__(verbose)
        self.beta = beta                        # intrinsic reward scaling factor
        self.lr = lr                            # learning rate for transition model
        self.batch_size = batch_size            # training batch size
        self.n_train_steps = n_train_steps      # training steps per update
        self.buffer_size = buffer_size          # maximum buffer size
        self.device = device

        # transition buffer: stores (state, action, next_state) tuples
        self.transition_buffer = []
        self.model = None
        self.optimizer = None
    
    def _on_training_start(self):
        """initialize the transition prediction model"""
        obs_space = self.training_env.observation_space
        action_space = self.training_env.action_space

        state_dim = int(np.prod(obs_space.shape))
        action_dim = int(np.prod(action_space.shape))
        hidden_dim = 64

        # reuse transition prediction model from CM score implementation
        self.model = TransitionPrediction(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        if self.verbose > 0:
            print(f"[LPM] initialized with state_dim={state_dim}, action_dim={action_dim}")
            print(f"[LPM] beta (learning progress scale): {self.beta}")
            print(f"[LPM] batch size: {self.batch_size}")
    
    def _compute_prediction_loss(self, states, actions, next_states):
        """compute MSE loss for transition prediction"""
        pred_next_states = self.model(states, actions)
        loss = nn.MSELoss()(pred_next_states, next_states)
        return loss

    def _get_batch_from_buffer(self):
        """sample a batch from the transition buffer"""
        if len(self.transition_buffer) < self.batch_size:
            return None, None, None

        indices = np.random.choice(len(self.transition_buffer), self.batch_size, replace=False)
        batch = [self.transition_buffer[i] for i in indices]

        states = torch.tensor(np.array([t[0] for t in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array([t[1] for t in batch]), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array([t[2] for t in batch]), dtype=torch.float32, device=self.device)
        
        return states, actions, next_states
    
    def _store_transitions(self, obs, actions):
        """store transitions in buffer (approximating next states)"""
        # flatten observations if needed
        obs_processed = obs.copy()
        if len(obs_processed.shape) > 2:
            obs_processed = obs_processed.reshape(obs_processed.shape, -1)
        
        actions_processed = actions.copy() if actions is not None else obs_processed
        if len(actions_processed.shape) > 2:
            actions_processed = actions_processed.reshape(actions_processed.shape, -1)
        
        # TODO: [not sure about this] for next_states, we approximate using current obs (suboptimal but workable)
        # we will need an environment wrapper to capture true next_states
        next_states_processed = obs_processed

        # store transitions
        for i in range(len(obs_processed)):
            self.transition_buffer.append((
                obs_processed[i],
                actions_processed[i] if actions is not None else obs_processed[i],
                next_states_processed[i]
            ))
        
        # maintain the buffer size
        if len(self.transition_buffer) > self.buffer_size:
            self.transition_buffer = self.transition_buffer[-self.buffer_size:]
    
    def _on_step(self):
        """called after each env step"""
        # access observations and rewards from PPO
        obs = self.locals.get('obs')
        actions = self.locals.get('actions')
        rewards = self.locals.get('rewards')

        if obs is None or rewards is None:
            return True

        # store new transitions
        self._store_transitions(obs, actions)

        # only compute learning progress if we have enough data
        if len(self.transition_buffer) < self.batch_size:
            return True

        # sample batch for learning progress calculation
        states, actions_batch, next_states = self._get_batch_from_buffer()
        if states is None:
            return True

        # compute prediction loss BEFORE training
        self.model.eval()
        with torch.no_grad():
            loss_before = self._compute_prediction_loss(states, actions_batch, next_states).item()
        
        # train the model for n_train_steps
        self.model.train()
        for _ in range(self.n_train_steps):
            self.optimizer.zero_grad()
            loss = self._compute_prediction_loss(states, actions_batch, next_states)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        # compute prediction loss AFTER training
        self.model.eval()
        with torch.no_grad():
            loss_after = self._compute_prediction_loss(states, actions_batch, next_states).item()

        # calculate learning progress: the reduction in prediction error
        learning_progress = loss_before - loss_after

        # scale learning progress and distribute across current batch
        intrinsic_reward = self.beta * learning_progress
        intrinsic_rewards = np.full_like(rewards, intrinsic_reward / len(rewards))

        # augment env rewards with learning progress rewards
        augmented_rewards = rewards + intrinsic_rewards
        self.locals['rewards'] = augmented_rewards

        if self.verbose > 0 and self.num_timesteps % 1000 == 0:
            print(f"[LPM] step {self.num_timesteps}: learning progress = {learning_progress:.6f}, intrinsic reward = {intrinsic_reward:.6f}")

        # log to wandb if available
        if wandb.run is not None and self.num_timesteps % 100 == 0:
            wandb.log({
                'lpm/learning_progress': learning_progress,
                'lpm/loss_before': loss_before,
                'lpm/loss_after': loss_after,
                'lpm/intrinsic_reward': intrinsic_reward,
                'step': self.num_timesteps
            })
        
        return True

# =====================
# Information Gain Intrinsic Reward Model
# =====================
class InfoRewardCallback(BaseCallback):
    """information gain intrinsic reward using ensemble disagreement"""
    def __init__(self, beta=1.0, lr=1e-3, update_freq=1000, batch_size=256, buffer_size=50000, device='cpu', verbose=0):
        super(InfoRewardCallback, self).__init__(verbose)
        self.beta = beta
        self.lr = lr
        self.update_freq = update_freq
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.device = device
        
        # model ensembles (reusing CM score logic)
        self.transition_models = []
        self.reward_models = []
        self.transition_optimizers = []
        self.reward_optimizers = []

        # data buffer: stores (state, action, reward, next_state) tuples
        self.data_buffer = []
        self.last_update_step = 0
    
    def _on_training_start(self):
        """initialize the ensemble models"""
        obs_space = self.training_env.observation_space
        action_space = self.training_env.action_space

        state_dim = int(np.prod(obs_space.shape))
        action_dim = int(np.prod(action_space.shape))
        hidden_dim = 64
        n_ensemble = 5

        # create ensembles of transition and reward models
        self.transition_models = [
            TransitionPrediction(state_dim, action_dim, hidden_dim).to(self.device)
            for _ in range(n_ensemble)
        ]
        self.reward_models = [
            RewardPrediction(state_dim, action_dim, hidden_dim).to(self.device)
            for _ in range(n_ensemble)
        ]

        # create the optimizers
        self.transition_optimizers = [
            optim.Adam(model.parameters(), lr=self.lr)
            for model in self.transition_models
        ]
        self.reward_optimizers = [
            optim.Adam(model.parameters(), lr=self.lr)
            for model in self.reward_models
        ]

        if self.verbose > 0:
            print(f"[Info] Initialized ensemble with {n_ensemble} models")
            print(f"[Info] State dim: {state_dim}, Action dim: {action_dim}")
            print(f"[Info] Beta (intrinsic reward scale): {self.beta}")
    
    def _store_transitions(self, obs, actions, rewards):
        """store transitions in buffer (approximating next_state for now)"""
        # TODO: you will need to modify this part

        # flatten observations if needed
        obs_processed = obs.copy()
        if len(obs_processed.shape) > 2:
            obs_processed = obs_processed.reshape(obs_processed.shape[0], -1)
        
        actions_processed = actions.copy() if actions is not None else obs_processed
        if len(actions_processed.shape) > 2:
            actions_processed.reshape(actions_processed.shape, -1)
        
        # TODO: this part below specifically...
        # for next states, we would approximate using current obs (suboptimal)
        # in practice, we need an env wrapper to capture true next_states
        next_states_processed = obs_processed   # PLACEHOLDER

        # store transitions
        for i in range(len(obs_processed)):
            self.data_buffer.append((
                obs_processed[i],
                actions_processed[i] if actions is not None else obs_processed[i],
                rewards[i],
                next_states_processed[i]
            ))
        
        # maintain the buffer size
        if len(self.data_buffer) > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
    
    def _get_batch_from_buffer(self):
        """sample a batch from the data buffer"""
        if len(self.data_buffer) < self.batch_size:
            return None, None, None, None
        
        indices = np.random.choice(len(self.data_buffer), self.batch_size, replace=False)
        batch = [self.data_buffer[i] for i in indices]

        states = torch.tensor(np.array([t[0] for t in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array([t[1] for t in batch]), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(np.array([t[2] for t in batch]), dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states = torch.tensor(np.array([t[3] for t in batch]), dtype=torch.float32, device=self.device)

        return states, actions, rewards, next_states
    
    def _train_ensemble_models(self, states, actions, rewards, next_states):
        """train the ensemble models on collected data"""
        transition_losses = []
        reward_losses = []

        # train transition models
        for model, optimizer in zip(self.transition_models, self.transition_optimizers):
            model.train()
            optimizer.zero_grad()
            pred_next_states = model(states, actions)
            loss = nn.MSELoss()(pred_next_states, next_states)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            transition_losses.append(loss.item())
        
        # train reward models
        for model, optimizer in zip(self.reward_models, self.reward_optimizers):
            model.train()
            optimizer.zero_grad()
            pred_rewards = model(states, actions)
            loss = nn.MSELoss()(pred_rewards, rewards)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            reward_losses.append(loss.item())
        
        return np.mean(transition_losses), np.mean(reward_losses)
    
    def _compute_ensemble_disagreement(self, states, actions):
        """compute disagreement (std dev) across ensemble predictions"""
        # compute transition model disagreement
        transition_preds = []
        for model in self.transition_models:
            model.eval()
            with torch.no_grad():
                pred = model(states, actions)
                transition_preds.append(pred)
        
        transition_stack = torch.stack(transition_preds)            # (n_ensemble, batch_size, state_dim)
        t_score = torch.std(transition_stack, dim=0).mean(dim=1)    # (batch_size,)

        # compute reward model disagreement
        reward_preds = []
        for model in self.reward_models:
            model.eval()
            with torch.no_grad():
                pred = model(states, actions)
                reward_preds.append(pred)
        
        reward_stack = torch.stack(reward_preds)            # (n_ensemble, batch_size, 1)
        r_score = torch.std(reward_stack, dim=0).squeeze()  # (batch_size,)
        
        return t_score.cpu().numpy(), r_score.cpu().numpy()
    
    def _on_step(self):
        """called after each environment step"""
        # access observations and rewards from PPO
        obs = self.locals.get('obs')
        actions = self.locals.get('actions')
        rewards = self.locals.get('rewards')

        if obs is None or rewards is None:
            return True
        
        # store new transitions
        self._store_transitions(obs, actions, rewards)

        # only compute intrinsic rewards if we have enough data
        if len(self.data_buffer) < self.batch_size:
            return True
        
        # periodically train ensemble models
        if (self.num_timesteps - self.last_update_step) >= self.update_freq:
            states, actions_batch, rewards_batch, next_states = self._get_batch_from_buffer()
            if states is not None:
                t_loss, r_loss = self._train_ensemble_models(states, actions_batch, rewards_batch, next_states)
                if self.verbose > 0:
                    print(f"[Info] Step {self.num_timesteps}: Transition loss={t_loss:.6f}, Reward loss={r_loss:.6f}")
                    self.last_update_step = self.num_timesteps
        
        # compute ensemble disagreement for current observations
        obs_processed = obs.copy()
        if len(obs_processed.shape) > 2:
            obs_processed = obs_processed.reshape(obs_processed.shape[0], -1)
        
        actions_processed = actions.copy() if actions is not None else obs_processed
        if len(actions_processed.shape) > 2:
            actions_processed = actions_processed.reshape(actions_processed.shape, -1)
        
        # convert to tensors
        obs_tensor = torch.tensor(obs_processed, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions_processed, dtype=torch.float32, device=self.device)

        # compute disagreement scores (information gain)
        t_scores, r_scores = self._compute_ensemble_disagreement(obs_tensor, actions_tensor)

        # calculate intrinsic rewards
        info_scores = t_scores + r_scores
        intrinsic_rewards = self.beta * info_scores

        # augment env rewards with intrinsic rewards
        augmented_rewards = rewards + intrinsic_rewards
        self.locals['rewards'] = augmented_rewards

        if self.verbose > 0 and self.num_timesteps % 1000 == 0:
            avg_intrinsic = np.mean(intrinsic_rewards)
            avg_t_score = np.mean(t_scores)
            avg_r_score = np.mean(r_scores)
            print(f"[Info] Step {self.num_timesteps}: Avg intrinsic reward = {avg_intrinsic:.4f} (t_score: {avg_t_score:.4f}, r_score: {avg_r_score:.4f})")
        
        # log to wandb if available
        if wandb.run is not None and self.num_timesteps % 100 == 0:
            wandb.log({
                'info/avg_intrinsic_reward': np.mean(intrinsic_rewards),
                'info/avg_transition_disagreement': np.mean(t_scores),
                'info/avg_reward_disagreement': np.mean(r_scores),
                'info/intrinsic_reward_std': np.std(intrinsic_rewards),
                'step': self.num_timesteps
            })
        
        return True

# =====================
# Info Training Function
# =====================
def train_info_baseline(args):
    """train PPO with info gain intrinsic rewards (no curriculum)"""
    logging.info("=== training Info baseline (no curriculum) ===")

    # load pretrained model
    if args.pretrained_path is None:
        args.pretrained_path = f'models/ppo_{args.task}_sb3/final_model.zip'
    
    set_random_seed(args.seed)
    student_model = PPO.load(args.pretrained_path)
    logging.info(f"[Info] using pretrained model: {args.pretrained_path}")

    # create base env (on interventions)
    def env_factory():
        return create_environment(
            args.task,
            intervention=None,      # no intervention for Info baseline
            seed=args.seed,
            skip_frame=args.skip_frame
        )
    
    # create vectorized env
    train_env = DummyVecEnv([env_factory])
    train_env = VecMonitor(train_env, filename=os.path.join(args.log_dir, 'info_monitor.csv'))
    
    # set up info callback
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    info_callback = InfoRewardCallback(
        beta=getattr(args, 'info_beta', 1.0),
        lr=getattr(args, 'info_lr', 1e-3),
        update_freq=getattr(args, 'info_update_freq', 1000),
        batch_size=getattr(args, 'info_batch_size', 256),
        device=device,
        verbose=1
    )

    # set up logging
    csv_logger = CSVLogger(args.log_dir)
    reward_monitor = RewardMonitorCallback("info_baseline", csv_logger, 0, 0)

    callback_list = CallbackList([
        info_callback,
        reward_monitor,
        WandbCallback(
            gradient_save_freq=100,
            model_save_path=args.log_dir if args.use_wandb else None,
            verbose=2
        ) if args.use_wandb else None
    ])
    callback_list.callbacks = [cb for cb in callback_list.callbacks if cb is not None]

    # set up SB3 logger
    sb3_log_path = os.path.join(args.log_dir, "sb3_csv_logs_info_baseline")
    new_logger = configure(sb3_log_path, ["stdout", "csv"])
    student_model.set_logger(new_logger)
    student_model.set_env(train_env)

    # train with information gain intrinsic rewards
    total_timesteps = args.timesteps * 7      # this is the same as the 7-stage curriculum
    logging.info(f"[Info] training for {total_timesteps} timesteps")

    start_time = time.time()
    student_model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        reset_num_timesteps=False
    )
    training_duration = time.time() - start_time

    logging.info(f"[Info] training completed in {training_duration:.2f} seconds")

    # save final model
    final_model_path = os.path.join(args.log_dir, "final_info_model.zip")
    student_model.save(final_model_path)
    logging.info(f"[Info] final model saved to {final_model_path}")

    # clean up
    train_env.close()

    return student_model

# =====================
# LPM Training Function
# =====================
def train_lpm_baseline(args):
    """train PPO with learning progress motivation; intrinsic rewards (no curriculum)"""
    logging.info("=== training LPM baseline (no curriculum) ===")

    # load pretrained model
    if args.pretrained_path is None:
        args.pretrained_path = f'models/ppo_{args.task}_sb3/final_model.zip'
    
    set_random_seed(args.seed)
    student_model = PPO.load(args.pretrained_path)
    logging.info(f"[LPM] using pretrained model: {args.pretrained_path}")

    # create base env (no interventions)
    def env_factory():
        return create_environment(
            args.task,
            intervention=None,      # no intervention for LPM baseline
            seed=args.seed,
            skip_frame=args.skip_frame
        )

    # create vectorized env
    train_env = DummyVecEnv([env_factory])
    train_env = VecMonitor(train_env, filename=os.path.join(args.log_dir, 'lpm_monitor.csv'))

    # set up LPM callback
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lpm_callback = LPMRewardCallback(
        beta=getattr(args, 'lpm_beta', 1.0),
        lr=getattr(args, 'lpm_lr', 1e-3),
        batch_size=getattr(args, 'lpm_batch_size', 256),
        n_train_steps=getattr(args, 'lpm_train_steps', 1),
        device=device,
        verbose=1
    )

    # set up logging
    csv_logger = CSVLogger(args.log_dir)
    reward_monitor = RewardMonitorCallback("lpm_baseline", csv_logger, 0, 0)

    callback_list = CallbackList([
        lpm_callback,
        reward_monitor,
        WandbCallback(
            gradient_save_freq=100,
            model_save_path=args.log_dir if args.use_wandb else None,
            verbose=2
        ) if args.use_wandb else None
    ])
    callback_list.callbacks = [cb for cb in callback_list.callbacks if cb is not None]

    # set up SB3 logger
    sb3_log_path = os.path.join(args.log_dir, "sb3_csv_logs_lpm_baseline")
    new_logger = configure(sb3_log_path, ["stdout", "csv"])
    student_model.set_logger(new_logger)
    student_model.set_env(train_env)

    # train with LPM intrinsic rewards
    total_timesteps = args.timesteps * 7      # this is the same as the 7-stage curriculum
    logging.info(f"[LPM] training for {total_timesteps} timesteps")

    start_time = time.time()
    student_model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        reset_num_timesteps=False
    )
    training_duration = time.time() - start_time

    logging.info(f"[LPM] training completed in {training_duration:.2f} seconds")

    # save final model
    final_model_path = os.path.join(args.log_dir, "final_lpm_model.zip")
    student_model.save(final_model_path)
    logging.info(f"[LPM] final model saved to {final_model_path}")

    # clean up
    train_env.close()

    return student_model

# =====================
# CSV Logger Class
# =====================
class CSVLogger:
    """handles csv logging for training metrics"""
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.csv_path = os.path.join(log_dir, 'training_log.csv')
        self.intervention_log_path = os.path.join(log_dir, 'intervention_log.csv')
        self.init_csv_files()
    
    def init_csv_files(self):
        """initialize csv files with headers"""
        # main training log
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'stage', 'intervention_type', 'episode', 'timestep',
                'reward', 'episode_length', 'success', 'cumulative_timesteps',
                'mean_reward_last_10', 'success_rate_last_10'
            ])
        
        # intervention selection log
        with open(self.intervention_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'stage', 'intervention_type', 'test_avg_reward', 'test_success_rate',
                'test_avg_length', 'selected', 'cumulative_timesteps', 'cm_score'
            ])
    
    def log_episode(self, stage, intervention_type, episode, timestep, reward,
                    episode_length, success, cumulative_timesteps, mean_reward_last_10, success_rate_last_10):
        """log episode data"""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                time.strftime('%Y-%m-%d %H:%M:%S'),
                stage, intervention_type, episode, timestep, reward,
                episode_length, success, cumulative_timesteps,
                mean_reward_last_10, success_rate_last_10
            ])
    
    def log_intervention_test(self, stage, intervention_type, test_avg_reward,
                              test_success_rate, test_avg_length, selected, cumulative_timesteps, cm_score=None):
        """log intervention test results"""
        with open(self.intervention_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                time.strftime('%Y-%m-%d %H:%M:%S'),
                stage, intervention_type, test_avg_reward, test_success_rate,
                test_avg_length, selected, cumulative_timesteps, cm_score
            ])

# =====================
# Visualization Class
# =====================
class TrainingVisualizer:
    """Simplified visualization class that uses enhanced system only"""
    def __init__(self, log_dir):
        self.log_dir = log_dir
        
    def plot_training_and_validation_curves(self, sb3_progress_path, validation_csv_path=None):
        """Generate training curve plots using enhanced visualization system"""
        try:
            from src.visualization import EnhancedTrainingVisualizer
            enhanced_visualizer = EnhancedTrainingVisualizer(self.log_dir)
            # Extract heuristic name from log directory, handling both replacement and non-replacement cases
            heuristic_name = os.path.basename(self.log_dir)
            heuristic_name = heuristic_name.replace('_replacement_sequencing_logs', '').replace('_sequencing_logs', '')
            enhanced_visualizer.plot_single_heuristic_analysis(
                sb3_progress_path, heuristic_name, validation_csv_path
            )
            logging.info("Enhanced visualization completed successfully")
        except ImportError:
            logging.warning("Enhanced visualization not available, skipping plot generation")
        except Exception as e:
            logging.error(f"Enhanced visualization failed: {e}")


class IntervenedCausalWorld:
    """
    A wrapper for CausalWorld that ensures a specific intervention is applied at the beginning of every episode.
    """
    def __init__(self, base_env: CausalWorld, intervention: dict):
        self.base_env = base_env
        self.intervention = intervention
        self.intervention_actor = intervention['class'](**intervention['params'])
        self.intervention_actor.initialize(self.base_env)
        self.reset_count = 0
        self.intervention_success_count = 0
        self.last_intervention_check = None

        # expose important attributes from the base environment
        self.observation_space = self.base_env.observation_space
        self.action_space = self.base_env.action_space

        print(f"IntervenedCausalWorld created with {(intervention['type'] if intervention is not None else 'none')} intervention")

    def reset(self, seed=0):
        """
        Resets the environment and then applies the intervention.
        """
        self.reset_count += 1
        # logging.info(f"[ENV RESET] Resetting environment (reset_count={self.reset_count})")
        obs = self.base_env.reset(seed=seed)
        try:
            variables_dict = self.base_env.get_variable_space_used()
            intervention_dict = self.intervention_actor._act(variables_dict)
            if intervention_dict:
                # the do_intervention method returns a tuple (success_signal, obs)
                success_signal, obs = self.base_env.do_intervention(intervention_dict)
                self.intervention_success_count += 1
                self.last_intervention_check = intervention_dict.copy()
                # log the first three resets
                if self.reset_count <= 3:
                    print(f"Reset #{self.reset_count}: {(self.intervention['type'] if self.intervention is not None else 'none')} intervention applied (success: {success_signal})")
            else:
                print(f"Reset #{self.reset_count}: No intervention dict for {(self.intervention['type'] if self.intervention is not None else 'none')}")
                self.last_intervention_check = None
        except Exception as e:
            logging.warning(f"Failed to apply intervention during reset: {e}")
            self.last_intervention_check = None
        return obs
    
    def step(self, action):
        """
        Steps the environment.
        """
        result = self.base_env.step(action)
        obs, reward, done, info = result
        # logging.info(f"[ENV STEP] Reward: {reward}, Done: {done}, Info: {info}")
        if done:
            # logging.info(f"[EPISODE END] Episode ended. Reward: {reward}, Info: {info}")
            pass # Early stopping: if 'success' in info and True, set done=True
        # Early stopping: if 'success' in info and True, set done=True
        if isinstance(info, dict) and 'success' in info and info['success']:
            done = True
            result = (obs, reward, done, info)
        return result
    
    def get_intervention_stats(self):
        return {
            'reset_count': self.reset_count,
            'intervention_success_count': self.intervention_success_count,
            'intervention_success_rate': self.intervention_success_count / max(self.reset_count, 1)
        }
    
    def __getattr__(self, name):
        """
        Delegate any other calls to the base environment.
        """
        return getattr(self.base_env, name)

class RewardMonitorCallback(BaseCallback):
    """callback to monitor training progress with detailed metrics"""
    def __init__(self, intervention_type="unknown", csv_logger=None, stage=0, cumulative_timesteps=0):
        super().__init__()
        self.intervention_type = intervention_type
        self.csv_logger = csv_logger
        self.stage = stage
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.step_count = 0
        self.episode_count = 0
        self.cumulative_timesteps = cumulative_timesteps

    def _on_step(self) -> bool:
        self.step_count += 1

        # check if we have infos and they contain episode data
        if (hasattr(self.locals, 'infos') and self.locals['info'] is not None and len(self.locals['infos']) > 0):
            for info in self.locals['infos']:
                if isinstance(info, dict) and 'episode' in info:
                    episode_info = info['episode']
                    self.episode_rewards.append(episode_info['r'])
                    self.episode_lengths.append(episode_info['l'])
                    self.episode_count += 1

                    # check for success if available
                    success = info.get('success', False)
                    self.episode_successes.append(success)

                    # calculate rolling averages
                    recent_rewards = self.episode_rewards[-10:]
                    recent_successes = self.episode_successes[-10:]
                    mean_reward_last_10 = np.mean(recent_rewards) if recent_rewards else 0
                    success_rate_last_10 = np.mean(recent_successes) if recent_successes else 0

                    # log to csv
                    if self.csv_logger:
                        self.csv_logger.log_episode(
                            self.stage, self.intervention_type, self.episode_count,
                            self.step_count, episode_info['r'], episode_info['l'],
                            success, self.cumulative_timesteps, mean_reward_last_10, success_rate_last_10
                        )
                    
                    # log recent episode completion
                    if len(self.episode_rewards) <= 5:  # log first 5 episodes
                        logging.info(f"[{self.intervention_type}] episode completed: reward={episode_info['r']:.3f}, length={episode_info['l']}")

                    # log to wandb if enabled
                    if wandb.run is not None:
                        wandb.log({
                            "stage": self.stage,
                            "intervention_type": self.intervention_type,
                            "episode_reward": episode_info['r'],
                            "episode_length": episode_info['l'],
                            "success": success,
                            "mean_reward_last_10": mean_reward_last_10,
                            "success_rate_last_10": success_rate_last_10,
                            "cumulative_timesteps": self.cumulative_timesteps + self.step_count
                        })

        return True

class InterventionLoggingCallback(BaseCallback):
    def __init__(self, intervention_type, stage, verbose=0):
        super().__init__(verbose)
        self.intervention_type = intervention_type
        self.stage = stage

    def _on_step(self) -> bool:
        # log custom context
        self.logger.record("custom/intervention_type", self.intervention_type)
        self.logger.record("custom/stage", self.stage)
        return True

class ValidationCallback(BaseCallback):
    """callback to evaluate on validation intervention set during training"""
    def __init__(self, validation_frequency=5000, task_name="pushing", csv_logger=None,
                 stage=0, cumulative_timesteps=0, validation_episodes=3, seed=0, verbose=0):
        super().__init__(verbose)
        self.validation_frequency = validation_frequency
        self.task_name = task_name
        self.csv_logger = csv_logger
        self.stage = stage
        self.cumulative_timesteps = cumulative_timesteps
        self.validation_episodes = validation_episodes
        self.seed = seed
        self.last_validation_step = 0
        self.validation_history = []    # storing the validation metrics over time
    
    def _on_step(self) -> bool:
        # is it time for validation?
        if (self.num_timesteps - self.last_validation_step) >= self.validation_frequency:
            self._evaluate_validation()
            self.last_validation_step = self.num_timesteps
        return True
    
    def _evaluate_validation(self):
        """evaluate model on validation intervention set"""
        logging.info(f"[VALIDATION] step {self.num_timesteps}: evaluating on validation set...")

        # create validation intervention
        validation_intervention = {
            "type": "validation",
            "class": ValidationInterventionActorPolicy,
            "params": {"seed": self.seed + self.num_timesteps}
        }

        # create validation environment
        validation_env = create_environment(
            self.task_name,
            validation_intervention,
            seed=self.seed + self.num_timesteps
        )

        total_reward = 0
        total_length = 0
        successes = 0
        episode_rewards = []

        for episode in range(self.validation_episodes):
            obs = validation_env.reset(seed=self.seed + self.num_timesteps + episode)
            done = False
            episode_reward = 0
            episode_length = 0
            episode_success = False

            while not done:
                # use current model for prediction
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = validation_env.step(action)
                episode_reward += reward
                episode_length += 1
        
        validation_env.close()

        # calculate metrics
        validation_metrics = {
            'validation_avg_reward': total_reward / self.validation_episodes,
            'validation_reward_std': np.std(episode_rewards),
            'validation_avg_length': total_length / self.validation_episodes,
            'validation_success_rate': successes / self.validation_episodes,
            'validation_step': self.num_timesteps,
            'validation_stage': self.stage
        }

        # store in history
        self.validation_history.append(validation_metrics)

        # log to console
        logging.info(f"[VALIDATION] reward: {validation_metrics['validation_avg_reward']:.3f}, Success: {validation_metrics['validation_success_rate']:.3f}")

        # log to CSV
        if self.csv_logger:
            self.csv_logger.log_validation_episode(
                self.stage, self.num_timesteps + self.cumulative_timesteps,
                validation_metrics['validation_avg_reward'],
                validation_metrics['validation_success_rate'],
                validation_metrics['validation_avg_length']
            )
        
        # log to WandB
        if wandb.run is not None:
            wandb.log({
                'validation/avg_reward': validation_metrics['validation_avg_reward'],
                'validation/success_rate': validation_metrics['validation_success_rate'],
                'validation/avg_length': validation_metrics['validation_avg_length'],
                'validation/reward_std': validation_metrics['validation_reward_std'],
                'stage': self.stage,
                'timesteps': self.num_timesteps + self.cumulative_timesteps
            })
        
        return validation_metrics

def create_environment(task_name, intervention=None, seed=0, skip_frame=3, max_episode_length=500):
    """create a causalworld environment with optional intervention"""
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
        seed=seed,
        max_episode_length=max_episode_length
    )

    if intervention is not None:
        return IntervenedCausalWorld(base_env, intervention)
    else:
        return base_env

def log_validation_episode(self, stage, timestep, validation_reward, validation_success_rate, validation_avg_length):
    """log validation episode data"""
    validation_csv_path = os.path.join(self.log_dir, 'validation_log.csv')

    # create file with headers if it does not exist
    if not os.path.exists(validation_csv_path):
        with open(validation_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'stage', 'timestep', 'validation_avg_reward',
                'validation_success_rate', 'validation_avg_length'
            ])
    
    # append validation data
    with open(validation_csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            time.strftime('%Y-%m-%d %H:%M:%S'),
            stage, timestep, validation_reward,
            validation_success_rate, validation_avg_length
        ])

def test_intervention_performance(student_model, intervention, task_name, num_episodes=10, seed=0):
    """test an intervention and return its average performance metrics"""
    set_seed(seed)
    logging.info(f"testing intervention: {(intervention['type'] if intervention is not None else 'none')}")
    
    # create environment with intervention
    # make sure test_intervention is applied to the same env config
    env = create_environment(task_name, intervention, seed=seed)

    total_reward = 0
    total_length = 0
    successes = 0
    episode_rewards = []

    for episode in range(num_episodes):
        obs = env.reset(seed=seed+episode)
        done = False
        episode_reward = 0
        episode_length = 0
        episode_success = False

        while not done:
            action, _ = student_model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1

            # check for success
            if isinstance(info, dict) and 'success' in info and info['success']:
                episode_success = True
        
        total_reward += episode_reward
        total_length += episode_length
        episode_rewards.append(episode_reward)
        if episode_success:
            successes += 1

        if episode < 3:     # log first 3 episodes
            logging.info(f"Episode {episode+1}: reward={episode_reward:.3f}, length={episode_length}, success={episode_success}")
        
    env.close()

    avg_reward = total_reward / num_episodes
    avg_length = total_length / num_episodes
    success_rate = successes / num_episodes
    reward_std = np.std(episode_rewards)

    metrics = {
        'avg_reward': avg_reward,
        'reward_std': reward_std,
        'avg_length': avg_length,
        'success_rate': success_rate,
        'total_episodes': num_episodes
    }

    logging.info(f"Results: avg_reward={avg_reward:.3f}, success_rate={success_rate:.3f}, avg_length={avg_length:.1f}")
    return metrics

# =====================
# RND Training Function
# =====================
def train_rnd_baseline(args):
    """train PPO with RND intrinsic rewards (no curriculum)"""
    logging.info("=== training RND baseline (no curriculum) ===")

    # load the pretrained model
    if args.pretrained_path is None:
        args.pretrained_path = f'models/ppo_{args.task}_sb3/final_model.zip'

    set_random_seed(args.seed)
    student_model = PPO.load(args.pretrained_path)
    logging.info(f"[RND] using pretrained model: {args.pretrained_path}")

    # create base environment (no interventions)
    def env_factory():
        return create_environment(
            args.task,
            intervention=None,
            seed=args.seed,
            skip_frame=args.skip_frame
        )
    
    # create vectorized env
    train_env = DummyVecEnv([env_factory])
    train_env = VecMonitor(train_env, filename=os.path.join(args.log_dir, 'rnd_monitor.csv'))

    # set up RND callback
    device = 'cuda'if torch.cuda.is_available() else 'cpu'
    rnd_callback = RNDIntrinsicRewardCallback(
        beta=getattr(args, 'rnd_beta', 0.01),
        update_freq=getattr(args, 'rnd_update_freq', 1000),
        batch_size=getattr(args, 'rnd_batch_size', 1024),
        device=device,
        verbose=1
    )

    # set up logging
    csv_logger = CSVLogger(args.log_dir)
    reward_monitor = RewardMonitorCallback("rnd_baseline", csv_logger, 0, 0)

    callback_list = CallbackList([
        rnd_callback,
        reward_monitor,
        WandbCallback(
            gradient_save_freq=100,
            model_save_path=args.log_dir if args.use_wandb else None,
            verbose=2
        ) if args.use_wandb else None
    ])
    callback_list.callbacks = [cb for cb in callback_list.callbacks if cb is not None]

    # set up SB3 logger
    sb3_log_path = os.path.join(args.log_dir, "sb3_csv_logs_rnd_baseline")
    new_logger = configure(sb3_log_path, ["stdout", "csv"])
    student_model.set_logger(new_logger)
    student_model.set_env(train_env)

    # train with RND intrinsic rewards
    total_timesteps = args.timesteps * 7    # same as the 7-stage curriculum
    logging.info(f"[RND] training for {total_timesteps} timesteps")

    start_time = time.time()
    student_model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        reset_num_timesteps=False
    )
    training_duration = time.time() - start_time

    logging.info(f"[RND] training completed {training_duration:.2f} seconds")

    # save the final model
    final_model_path = os.path.join(args.log_dir, "final_rnd_model.zip")
    student_model.save(final_model_path)
    logging.info(f"[RND] Final model saved to {final_model_path}")

    # clean up
    train_env.close()

    return student_model

# =====================
# Count-based Exploration Model
# =====================
class CountBasedRewardCallback(BaseCallback):
    """callback that augments env rewards with count-based intrinsic rewards"""
    def __init__(self, beta=0.01, encoding_dim=32, verbose=0):
        super(CountBasedRewardCallback, self).__init__(verbose)
        self.beta = beta                    # intrinsic reward scaling factor
        self.encoding_dim = encoding_dim    # dimension for state hashing
        self.visit_counts = {}              # dictionary to store state visit counts
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.projection_matrix = None
        self.total_states_visited = 0
    
    def _on_training_start(self):
        """initialize the random projection matrix for state hashing"""
        obs_space = self.training_env.observation_space
        obs_dim = int(np.prod(obs_space.shape))

        # create fixed random projection matrix for state hashing
        self.projection_matrix = torch.randn(obs_dim, self.encoding_dim, device=self.device)

        if self.verbose > 0:
            print(f"[CountBased] Initialized with observation dim: {obs_dim}")
            print(f"[CountBased] Projection matrix-based shape: {self.projection_matrix.shape}")
            print(f"[CountBased] Beta (intrinsic reward scale): {self.beta}")
    
    def _hash_state(self, obs_tensor):
        """
        hash continuous state vectors into discrete codes using random projection

        args:
            obs_tensor: torch tensor of shape (batch_size, obs_dim)
        
        returns:
            list of hashable tuples representing discrete state codes
        """
        with torch.no_grad():
            # project to lower dimension
            projected = obs_tensor @ self.projection_matrix     # (batch_size, encoding_dim)

            # binarize by checking if > 0
            binary_codes = (projected > 0).cpu().numpy()

            # convert to hashable tuples
            if binary_codes.ndim == 1:
                # single state
                return [tuple(binary_codes.astype(np.int8))]
            else:
                # batch of states
                return [tuple(code.astype(np.int8)) for code in binary_codes]
    
    def _on_step(self):
        """called after each environment step"""
        # access observations and rewards from PPO
        obs = self.locals.get('obs')
        rewards = self.locals.get('rewards')

        if obs is None or rewards is None:
            return True

        # flatten observations if needed
        obs_processed = obs.copy()
        if len(obs_processed.shape) > 2:
            obs_processed = obs_processed.reshape(obs_processed.shape[0], -1)
        
        # convert to tensor
        obs_tensor = torch.tensor(obs_processed, dtype=torch.float32, device=self.device)

        # hash states to get discrete codes
        state_hashes = self._hash_state(obs_tensor)

        # calculate intrinsic rewards based on visit counts
        intrinsic_rewards = []
        for state_hash in state_hashes:
            # update visit count
            current_count = self.visit_counts.get(state_hash, 0)
            self.visit_counts[state_hash] = current_count + 1

            # calculate intrinsic reward: beta / sqrt(count)
            intrinsic_reward = self.beta / np.sqrt(self.visit_counts[state_hash])
            intrinsic_rewards.append(intrinsic_reward)
        
        intrinsic_rewards = np.array(intrinsic_rewards)

        # augment env rewards with intrinsic rewards
        augmented_rewards = rewards + intrinsic_rewards

        # update rewards in-place (affects PPO's learning)
        self.locals['rewards'] = augmented_rewards

        # update statistics
        self.total_states_visited += len(state_hashes)

        if self.verbose > 0 and self.num_timesteps % 1000 == 0:
            unique_states = len(self.visit_counts)
            avg_intrinsic = np.mean(intrinsic_rewards)
            print(f"[CountBased] Step {self.num_timesteps}: Unique states visited: {unique_states}, Avg intrinsic reward: {avg_intrinsic:.4f}")
        
        # log to wandb if possible
        if wandb.run is not None and self.num_timesteps % 100 == 0:
            wandb.log({
                'count_based/unique_states': len(self.visit_counts),
                'count_based/avg_intrinsic_reward': np.mean(intrinsic_rewards),
                'count_based/intrinsic_reward_std': np.std(intrinsic_rewards),
                'step': self.num_timesteps
            })
        
        return True

# =====================
# Count-based Training Function
# =====================
def train_count_baseline(args):
    """train PPO with count-based intrinsic rewards (no curriculum)"""
    logging.info(f"=== training count-based baseline (no curriculum) ===")

    # load pretrained model
    if args.pretrained_path is None:
        args.pretrained_path = f'models/ppo_{args.task}_sb3/final_model.zip'

    set_random_seed(args.seed)
    student_model = PPO.load(args.pretrained_path)
    logging.info(f"[CountBased] Using pretrained model: {args.pretrained_path}")

    # create base environment (no interventions)
    def env_factory():
        return create_environment(
            args.task,
            intervention=None,      # no intervention for count-based baseline
            seed=args.seed,
            skip_frame=args.skip_frame
        )
    
    # create vectorized env
    train_env = DummyVecEnv([env_factory])
    train_env = VecMonitor(train_env, filename=os.path.join(args.log_dir, 'count_monitor.csv'))

    # set up count-based callback
    count_callback = CountBasedRewardCallback(
        beta=getattr(args, 'count_beta', 0.01),
        encoding_dim=getattr(args, 'count_encoding_dim', 32),
        verbose=1
    )

    # set up logging
    csv_logger = CSVLogger(args.log_dir)
    reward_monitor = RewardMonitorCallback("count_baseline", csv_logger, 0, 0)

    callback_list = CallbackList([
        count_callback,
        reward_monitor,
        WandbCallback(
            gradient_save_freq=100,
            model_save_path=args.log_dir if args.use_wandb else None,
            verbose=2
        ) if args.use_wandb else None
    ])
    callback_list.callbacks = [cb for cb in callback_list.callbacks if cb is not None]

    # set up SB3 logger
    sb3_log_path = os.path.join(args.log_dir, "sb3_csv_logs_count_baseline")
    new_logger = configure(sb3_log_path, ["stdout", "csv"])
    student_model.set_logger(new_logger)
    student_model.set_env(train_env)

    # train with count-based intrinsic rewards
    total_timesteps = args.timesteps * 7    # same total as 7-stage curriculum
    logging.info(f"[CountBased] Training for {total_timesteps} timesteps")

    start_time = time.time()
    student_model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        reset_num_timesteps=False
    )
    training_duration = time.time() - start_time

    logging.info(f"[CountBased] Training completed in {training_duration:.2f} seconds")

    # log final statistics
    unique_states = len(count_callback.visit_counts)
    logging.info(f"[CountBased] Final unique states visited: {unique_states}")

    # save final model
    final_model_path = os.path.join(args.log_dir, "final_count_model.zip")
    student_model.save(final_model_path)
    logging.info(f"[CountBased] Final model saved to {final_model_path}")

    # clean up
    train_env.close()

    return student_model, count_callback

def train_on_intervention(student_model, intervention, task_name, timesteps, args, stage_num, total_stages, csv_logger, cumulative_timesteps):
    """train the student model on a specific intervention"""
    type = intervention['type'] if intervention is not None else 'none'
    logging.info(f"=== stage {stage_num}/{total_stages}: training on {type} intervention ===")

    # set up sb3 logger to log to CSV
    sb3_log_path = os.path.join(args.log_dir, f"sb3_csv_logs_{stage_num}_{type}")
    new_logger = configure(sb3_log_path, ["stdout", "csv"])
    student_model.set_logger(new_logger)

    # create environment factory for training
    def env_factory():
        env_idx = 0    # would modify this for each env in a parallel setup
        return create_environment(
            task_name,
            intervention,
            seed=args.seed + stage_num * 1000 + env_idx,
            skip_frame=args.skip_frame
        )
    
    # create vectorized environment
    train_env = DummyVecEnv([env_factory])
    train_env = VecMonitor(train_env, filename=os.path.join(sb3_log_path, f'monitor.csv'))

    # add validation callback
    validation_callback = ValidationCallback(
        validation_frequency=getattr(args, 'validation_frequency', 5000),   # default every 5000 steps
        task_name=task_name,
        csv_logger=csv_logger,
        stage=stage_num,
        cumulative_timesteps=cumulative_timesteps,
        validation_episodes=getattr(args, 'validation_episodes', 10),   # default is 10 episodes
        seed=args.seed + stage_num * 10000  # unique seed for validation
    )

    # set up callback for monitoring
    reward_monitor = RewardMonitorCallback(type, csv_logger, stage_num, cumulative_timesteps)
    callback_list = CallbackList([
        reward_monitor,
        validation_callback,
        InterventionLoggingCallback(type, stage_num),
        WandbCallback(
            gradient_save_freq=100,
            model_save_path=args.log_dir if args.use_wandb else None,
            verbose=2
        ) if args.use_wandb else None
    ])
    # removing none if wandb is not used
    callback_list.callbacks = [cb for cb in callback_list.callbacks if cb is not None]

    # set the new environment
    student_model.set_env(train_env)

    # train the model
    start_time = time.time()
    student_model.learn(
        total_timesteps=timesteps,
        callback=callback_list,
        reset_num_timesteps=False
    )
    training_duration = time.time() - start_time

    # save model after training
    model_path = os.path.join(args.log_dir, f"model_stage_{stage_num}_{type}")
    student_model.save(model_path)
    logging.info(f"model saved to {model_path}")

    # clean up
    train_env.close()

    return reward_monitor.cumulative_timesteps, validation_callback.validation_history

def evaluate_final_performance(student_model, task_name, num_episodes=20, seed=0):
    set_seed(seed)
    """evaluate final performance on validation environment (no interventions)"""
    logging.info("===final evaluation===")

    # create validation environment
    validation_env = create_environment(task_name, intervention=None, seed=seed)

    total_reward = 0
    total_length = 0
    successes = 0
    episode_rewards = []

    for episode in range(num_episodes):
        obs = validation_env.reset(seed=seed+episode)
        done = False
        episode_reward = 0
        episode_length = 0
        episode_success = False

        while not done:
            action, _ = student_model.predict(obs, deterministic=True)
            obs, reward, done, info = validation_env.step(action)
            episode_reward += reward
            episode_length += 1

            if isinstance(info, dict) and 'success' in info and info['success']:
                episode_success = True
        
        total_reward += episode_reward
        total_length += episode_length
        episode_rewards.append(episode_reward)
        if episode_success:
            successes += 1
        
        if episode >= 15:  # log the last 5 episodes
            logging.info(f"episode {episode+1}: reward={episode_reward:.3f}, length={episode_length}, success={episode_success}")
        
    validation_env.close()

    final_metrics = {
        'avg_reward': total_reward / num_episodes,
        'reward_std': np.std(episode_rewards),
        'avg_length': total_length / num_episodes,
        'success_rate': successes / num_episodes,
        'total_episodes': num_episodes
    }

    logging.info(f"Final performance")
    logging.info(f"average reward: {final_metrics['avg_reward']:.3f} +/- {final_metrics['reward_std']:.3f}")
    logging.info(f"success rate: {final_metrics['success_rate']:.3f}")
    logging.info(f"average episode length: {final_metrics['avg_length']:.1f}")

    return final_metrics

def aggregate_sb3_progress(log_dir):
    import glob
    import logging
    import pandas as pd
    all_dfs = []
    for path in glob.glob(os.path.join(log_dir, "sb3_csv_logs_*", "progress.csv")):
        dirname = os.path.basename(os.path.dirname(path))
        parts = dirname.split('_')
        stage = parts[3] if len(parts) > 3 else 'unknown'
        intervention = parts[4] if len(parts) > 4 else 'unknown'
        df = pd.read_csv(path)
        df['stage'] = stage
        df['intervention_type'] = intervention
        all_dfs.append(df)
    if not all_dfs:
        print("No progress.csv files found.")
        return None
    all_progress = pd.concat(all_dfs, ignore_index=True)
    rename_map = {
        'rollout/ep_rew_mean': 'mean_episode_reward',
        'time/time_elapsed': 'elapsed_time',
        'time/iterations': 'iteration',
        'rollout/ep_len_mean': 'mean_episode_length',
        'time/total_timesteps': 'total_timesteps',
        'custom/intervention_type': 'intervention_type',
        'custom/stage': 'stage',
        'time/fps': 'frames_per_second',
        'train/explained_variance': 'explained_variance',
        'train/value_loss': 'value_loss',
        'train/loss': 'policy_loss',
        'train/clip_fraction': 'clip_fraction',
        'train/entropy_loss': 'entropy_loss',
        'train/learning_rate': 'learning_rate',
        'train/std': 'policy_std',
        'train/clip_range': 'clip_range',
        'train/n_updates': 'num_updates',
        'train/policy_gradient_loss': 'policy_gradient_loss',
        'train/approx_kl': 'approx_kl_divergence',
    }
    all_progress.rename(columns=rename_map, inplace=True)
    # Sanity check for expected columns
    expected_cols = set(rename_map.values())
    missing_cols = expected_cols - set(all_progress.columns)
    if missing_cols:
        logging.warning(f"Missing expected columns in SB3 progress aggregation: {missing_cols}")
    out_path = os.path.join(log_dir, "all_progress.csv")
    all_progress.to_csv(out_path, index=False)
    print(f"Aggregated SB3 progress saved to {out_path}")
    return out_path

# ==================
# Sequential pacing signal loop
# ==================
def main():
    parser = argparse.ArgumentParser(description="Baselines for curriculum learning in CausalWorld")
    parser.add_argument('--curriculum_mode', type=str, default='greedy',
                        choices=['greedy', 'random', 'none', 'cm', 'rnd', 'count', 'lpm', 'info', 'autocalc'])
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--eval', action='store_true', help='Evaluate the model')
    parser.add_argument('--task', type=str, default='pushing', help='Task name')
    parser.add_argument('--timesteps', type=int, default=50000, help='Timesteps for each intervention training block')
    parser.add_argument('--pretrained_path', type=str, help='Path to pretrained PPO model')
    parser.add_argument('--log_dir', type=str, default=None, help='Log directory (will be auto-generated if not specified)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--skip_frame', type=int, default=3, help='Frame skip')
    parser.add_argument('--use_wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--test_episodes', type=int, default=10, help='Episodes for testing each intervention')
    parser.add_argument('--alpha_cm', type=float, default=0.5, help='Alpha parameter for causal mismatch')
    parser.add_argument('--validation_frequency', type=int, default=5000, help='Steps between validation evaluations')
    parser.add_argument('--validation_episodes', type=int, default=10, help='Episodes per validation evaluation')
    parser.add_argument('--rnd_beta', type=float, default=0.01, help='RND intrinsic reward scaling factor')
    parser.add_argument('--rnd_update_freq', type=int, default=1000, help='RND predictor update frequency')
    parser.add_argument('--rnd_batch_size', type=int, default=1024, help='RND predictor training batch size')
    parser.add_argument('--count_beta', type=float, default=0.01, help='Count-based intrinsic reward scaling factor')
    parser.add_argument('--count_encoding_dim', type=int, default=32, help='Encoding dimension for state hashing')
    parser.add_argument('--lpm_beta', type=float, default=1.0, help='LPM reward scaling factor')
    parser.add_argument('--lpm_lr', type=float, default=1e-3, help='LPM transition model learning rate')
    parser.add_argument('--lpm_batch_size', type=int, default=256, help='LPM training batch size')
    parser.add_argument('--lpm_train_steps', type=int, default=1, help='LPM model training steps per update')
    parser.add_argument('--info_beta', type=float, default=1.0, help='Info gain intrinsic reward scaling factor')
    parser.add_argument('--info_lr', type=float, default=1e-3, help='Info ensemble model learning rate')
    parser.add_argument('--info_update_freq', type=int, default=1000, help='Info ensemble update frequency')
    parser.add_argument('--info_batch_size', type=int, default=256, help='Info ensemble training batch size')
    parser.add_argument('--replacement', action='store_true', help='Allow intervention replacement (same intervention can be selected multiple times)')

    args = parser.parse_args()

    # Only set log directory automatically if not provided by user
    if args.log_dir is None:
        # Create log directory name with replacement indicator if applicable
        base_log_name = f"{args.curriculum_mode}_sequencing_logs"
        
        # Add replacement indicator for intervention-based methods
        if args.replacement and args.curriculum_mode in ['random', 'greedy', 'cm']:
            base_log_name = f"{args.curriculum_mode}_replacement_sequencing_logs"
        
        # Place all logs in a centralized 'logs' directory
        args.log_dir = os.path.join('logs', base_log_name)

    
    if args.task not in TASK_BENCHMARKS:
        import sys
        print(f"\nERROR: task '{args.task}' is not supported for evaluation.")
        print(f"supported tasks: {SUPPORTED_TASKS}")
        sys.exit(1)

    if args.train:
        # set random seed
        set_seed(args.seed)

        # create log directory
        os.makedirs(args.log_dir, exist_ok=True)
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(process)d %(levelname)s %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(args.log_dir, 'seq_pacing_signal.log')),
                logging.StreamHandler()
            ]
        )
        # initialize the csv logger and visualizer
        csv_logger = CSVLogger(args.log_dir)
        visualizer = TrainingVisualizer(args.log_dir)

        # initialize wandb if requested
        if args.use_wandb:
            wandb.init(
                project=f'curriculum-sequencing-{args.task}',
                name=f'{args.curriculum_mode}_{args.task}_seed{args.seed}',
                config=vars(args),
                tags=[args.task, args.curriculum_mode, 'curriculum', 'sequencing']
            )

        if args.curriculum_mode == 'rnd':
            logging.info(f"Starting RND baseline training (no curriculum) with args: {args}")
            # Train RND baseline (no curriculum)
            final_model = train_rnd_baseline(args)
            
            # Evaluate final performance
            final_performance = evaluate_final_performance(final_model, args.task, num_episodes=20, seed=args.seed + 999)
            logging.info(f"RND baseline final performance: {final_performance}")
            
            # Generate plots
            sb3_progress_path = aggregate_sb3_progress(args.log_dir)
            if sb3_progress_path:
                visualizer.plot_training_and_validation_curves(sb3_progress_path)  # No validation curves for RND baseline
            
            # Save results
            results_path = os.path.join(args.log_dir, "rnd_results.json")
            with open(results_path, 'w') as f:
                json.dump({
                    'final_performance': final_performance,
                    'args': vars(args)
                }, f, indent=2)
            logging.info(f"Results saved to {results_path}")
            
            if args.use_wandb:
                wandb.log({
                    'final_reward': final_performance['avg_reward'],
                    'final_success_rate': final_performance['success_rate'],
                    'total_timesteps': args.timesteps * 7  # same as curriculum
                })
                wandb.finish()
            return
        
        elif args.curriculum_mode == 'count':
            logging.info(f"Starting count-based baseline training (no training) with args: {args}")

            # training count-based baseline (no curriculum)
            final_model, count_callback = train_count_baseline(args)

            # evaluate final performance
            final_performance = evaluate_final_performance(final_model, args.task, num_episodes=20, seed=args.seed)
            logging.info(f"Count-based baseline final performance: {final_performance}")

            # generate plots
            sb3_progress_path = aggregate_sb3_progress(args.log_dir)
            if sb3_progress_path:
                visualizer.plot_training_and_validation_curves(sb3_progress_path)   # no validation curves for count baseline

            # save results
            results_path = os.path.join(args.log_dir, "count_results.json")
            with open(results_path, 'w') as f:
                json.dump({
                    'final_performance': final_performance,
                    'unique_states_visited': len(count_callback.visit_counts),
                    'args': vars(args)
                }, f, indent=2)
            logging.info(f"Results saved to {results_path}")
            
            if args.use_wandb:
                wandb.log({
                    'final_reward': final_performance['avg_reward'],
                    'final_success_rate': final_performance['success_rate'],
                    'unique_states_visited': len(count_callback.visit_counts),
                    'total_timesteps': args.timesteps * 7
                })
                wandb.finish()

            return
        
        elif args.curriculum_mode == 'lpm':
            logging.info(f"Starting LPM baseline training (no curriculum) with args: {args}")

            # train LPM baseline (no curriculum)
            final_model = train_lpm_baseline(args)

            # evaluate final performance
            final_performance = evaluate_final_performance(final_model, args.task, num_episodes=20, seed=args.seed)
            logging.info(f"LPM baseline final performance: {final_performance}")

            # generate plots
            sb3_progress_path = aggregate_sb3_progress(args.log_dir)
            if sb3_progress_path:
                visualizer.plot_training_and_validation_curves(sb3_progress_path)
            
            # save results
            results_path = os.path.join(args.log_dir, "lpm_results.json")
            with open(results_path, 'w') as f:
                json.dump({
                    'final_performance': final_performance,
                    'args': vars(args)
                }, f, indent=2)
            logging.info(f"Results saved to {results_path}")

            if args.use_wandb:
                wandb.log({
                    'final_reward': final_performance['avg_reward'],
                    'final_success_rate': final_performance['success_rate'],
                    'total_timesteps': args.timesteps * 7
                })
                wandb.finish()

            return

        elif args.curriculum_mode == 'info':
            logging.info(f"Starting Info baseline training (no curriculum) with args: {args}")

            # train info baseline (no curriculum)
            final_model = train_info_baseline(args)

            # evaluate final performance
            final_performance = evaluate_final_performance(final_model, args.task, num_episodes=20, seed=args.seed)
            logging.info(f"Info baseline final performance: {final_performance}")

            # generate the plots
            sb3_progress_path = aggregate_sb3_progress(args.log_dir)
            if sb3_progress_path:
                visualizer.plot_training_and_validation_curves(sb3_progress_path)
            
            # save results
            results_path = os.path.join(args.log_dir, "info_results.json")
            with open(results_path, 'w') as f:
                json.dump({
                    'final_performance': final_performance,
                    'args': vars(args)
                }, f, indent=2)
            logging.info(f"Results saved to {results_path}")

            if args.use_wandb:
                wandb.log({
                    'final_reward': final_performance['avg_reward'],
                    'final_success_rate': final_performance['success_rate'],
                    'total_timesteps': args.timesteps * 7
                })
                wandb.finish()

            return

        else:
            logging.info(f"Starting curriculum sequencing logic with args: {args}")

        if args.pretrained_path is None:
            args.pretrained_path = f'models/ppo_{args.task}_sb3/final_model.zip'
        set_random_seed(args.seed)  # seed before loading the model
        student_model = PPO.load(args.pretrained_path)
        logging.info(f"[PRETRAINED] Using pretrained model path: {args.pretrained_path}")

        # our tracking variables
        remaining_interventions = INTERVENTIONS.copy()
        completed_sequence = []
        all_results = []
        cumulative_timesteps = 0

        logging.info(f"Starting with {len(remaining_interventions)} interventions")
        logging.info(f"Replacement mode: {args.replacement}")

        # evaluate initial performance
        initial_performance = evaluate_final_performance(student_model, args.task, num_episodes=10, seed=args.seed)
        logging.info(f"initial performance: {initial_performance}")
        
        # main sequencing loop
        stage = 1
        total_stages = 7  # Always run for 7 stages for fair comparison

        while stage <= total_stages:
            logging.info(f"CURRICULUM STAGE {stage}/{total_stages}")
            if args.replacement:
                # With replacement: always have all interventions available
                available_interventions = INTERVENTIONS.copy()
                logging.info(f"Available interventions (with replacement): {[i['type'] for i in available_interventions]}")
            else:
                # Without replacement: use remaining interventions
                available_interventions = remaining_interventions
                logging.info(f"Remaining interventions: {[i['type'] for i in available_interventions]}")

            if args.curriculum_mode == 'none':
                # no interventions, no testing, just train on the base environment
                logging.info(f"CURRICULUM STAGE {stage}/{total_stages} (no interventions)")
                intervention_to_train = None    # no intervention
                test_metrics = None

                # train on the base environment (no intervention)
                cumulative_timesteps, validation_history = train_on_intervention(
                    student_model,
                    intervention_to_train,  # this should be handled in the create_env logic
                    args.task,
                    args.timesteps,
                    args,
                    stage,
                    total_stages,
                    csv_logger,
                    cumulative_timesteps
                )

                completed_sequence.append({
                    'stage': stage,
                    'intervention': 'none',
                    'test_metrics': test_metrics
                })

                logging.info(f"\nCompleted stage {stage}. No intervention used.")
                logging.info(f"Stages remaining: {total_stages - stage}")

                if args.use_wandb:
                    wandb.log({
                        f'stage_{stage}_intervention': 'none',
                        f'stage_{stage}_test_reward': None,
                        f'stage_{stage}_test_success': None,
                        'stage': stage,
                        'remaining_interventions': total_stages - stage,
                        'cumulative_timesteps': cumulative_timesteps 
                    })
                stage += 1
                continue  # Skip the rest of the loop for 'none' mode
            elif args.curriculum_mode == 'random':
                # random mode: pick one at random, no testing
                set_seed(args.seed + stage * 100)
                selected_intervention = random.choice(available_interventions)
                logging.info(f"Randomly selected intervention: {selected_intervention['type']}")
                # log to CSV with test fields as None
                csv_logger.log_intervention_test(
                    stage, selected_intervention['type'], None, None, None, True, cumulative_timesteps, cm_score=None
                )
                intervention_to_train = selected_intervention
                test_metrics = None
            elif args.curriculum_mode == 'greedy':
                # 1. test all available interventions
                intervention_results = []
                for i, intervention in enumerate(available_interventions):
                    logging.info(f"\nTesting intervention {i+1}/{len(available_interventions)}: {intervention['type']}")

                    metrics = test_intervention_performance(
                        student_model,
                        intervention,
                        args.task,
                        num_episodes=args.test_episodes,
                        seed=args.seed + stage * 100
                    )

                    intervention_results.append({
                        'intervention': intervention,
                        'metrics': metrics,
                        'avg_reward': metrics['avg_reward']
                    })

                    # log intervention test to csv
                    csv_logger.log_intervention_test(
                        stage, intervention['type'], metrics['avg_reward'], metrics['success_rate'],
                        metrics['avg_length'], False, cumulative_timesteps, cm_score=None
                    )
                
                # 2. find intervention with the highest average reward
                best_result = max(intervention_results, key=lambda x: x['avg_reward'])
                intervention_to_train = best_result['intervention']
                best_metrics = best_result['metrics']
                test_metrics = best_metrics

                logging.info(f"best intervention for stage {stage}: {intervention_to_train['type']}")
                logging.info(f"average reward: {best_metrics['avg_reward']:.3f}")
                logging.info(f"success rate: {best_metrics['success_rate']:.3f}")
                logging.info(f"average length: {best_metrics['avg_length']:.1f}")
            elif args.curriculum_mode == 'cm':
                # incorporate the causal-mismatch logic (Cho et al: https://www.arxiv.org/abs/2507.02910) in here
                # 1. test all available interventions by evaluating their CM score
                intervention_results = []
                for i, intervention in enumerate(available_interventions):
                    logging.info(f"\nTesting intervention {i+1}/{len(available_interventions)}: {intervention['type']} (CM score)")
                    # create environment with intervention
                    env = create_environment(
                        args.task,
                        intervention,
                        seed=args.seed + stage * 100
                    )
                    # evaluate CM score
                    cm_score = evaluate_cm_score(
                        env,
                        student_model,
                        max_episodes=args.test_episodes,
                        max_episode_length=500,
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        intervention_type=intervention['type'],
                        seed=args.seed + stage * 100
                    )

                    metrics = test_intervention_performance(
                        student_model,
                        intervention,
                        args.task,
                        num_episodes=args.test_episodes,
                        seed=args.seed
                    )

                    intervention_results.append({
                        'intervention': intervention,
                        'cm_score': cm_score,
                        'avg_reward': metrics['avg_reward']
                    })
                    # logging to csv
                    csv_logger.log_intervention_test(
                        stage, intervention['type'], metrics['avg_reward'], metrics['success_rate'],
                        metrics['avg_length'], False, cumulative_timesteps, cm_score=cm_score
                    )
                    env.close()
                
                cm_scores = [r['cm_score'] for r in intervention_results]
                rewards = [r['avg_reward'] for r in intervention_results]

                cm_min, cm_max = min(cm_scores), max(cm_scores)
                reward_min, reward_max = min(rewards), max(rewards)
                
                for result in intervention_results:
                    # normalizing cm score (higher = unfamiliar)
                    norm_cm = (result['cm_score'] - cm_min) / (cm_max - cm_min + 1e-8)
                    
                    # normalizing reward (higher = learnable)
                    norm_reward = (result['avg_reward'] - reward_min) / (reward_max - reward_min + 1e-8)

                    # combined objective: balance novelty and learnability
                    # NOTE: the alpha weight can be experimented with
                    alpha = args.alpha_cm
                    result['unified_score'] = alpha * norm_cm + (1 - alpha) * norm_reward
                    logging.info(
                        f"[Stage {stage}] {result['intervention']['type']:<12} | "
                        f"reward={result['avg_reward']:.2f} (norm {norm_reward:.3f}) | "
                        f"CM={result['cm_score']:.3f} (norm {norm_cm:.3f}) | "
                        f"α={alpha:.2f} → unified={result['unified_score']:.3f}"
                    )
                # 2. Pick the intervention with the highest CM + reward score
                best_result = max(intervention_results, key=lambda x: x['unified_score'])
                intervention_to_train = best_result['intervention']
                test_metrics = {
                    'avg_reward': best_result['avg_reward'],
                    'cm_score': best_result['cm_score'],
                    'unified_score': best_result['unified_score']
                }
                logging.info(f"Best intervention for stage {stage}: {intervention_to_train['type']} (Unified score: {best_result['unified_score']:.4f})")

            
            # 3. train on the best intervention
            cumulative_timesteps, validation_history = train_on_intervention(
                student_model,
                intervention_to_train,
                args.task,
                args.timesteps,
                args,
                stage,
                total_stages,
                csv_logger,
                cumulative_timesteps
            )

            # 4. remove the intervention from the remaining list (only if replacement=False)
            if not args.replacement and intervention_to_train is not None:
                remaining_interventions.remove(intervention_to_train)
                removal_msg = "removed from list"
            else:
                removal_msg = "kept available for future selection" if args.replacement else "no intervention to remove"
            
            completed_sequence.append({
                'stage': stage,
                'intervention': (intervention_to_train['type'] if intervention_to_train is not None else 'none'),
                'test_metrics': test_metrics
            })

            # log progress
            logging.info(f"\nCompleted stage {stage}. Intervention '{(intervention_to_train['type'] if intervention_to_train is not None else 'none')}' {removal_msg}.")
            if args.replacement:
                logging.info(f"Replacement mode: All interventions remain available")
            else:
                logging.info(f"Remaining interventions: {len(remaining_interventions)}")
            
            # WandB logging
            if args.use_wandb:
                wandb.log({
                    f'stage_{stage}_intervention': (intervention_to_train['type'] if intervention_to_train is not None else 'none'),
                    f'stage_{stage}_test_reward': test_metrics.get('avg_reward') if test_metrics else None,
                    f'stage_{stage}_test_success': test_metrics.get('success_rate') if test_metrics else None,
                    f'stage_{stage}_cm_score': test_metrics.get('cm_score') if test_metrics else None,
                    f'stage_{stage}_rnd_score': test_metrics.get('rnd_score') if test_metrics else None,
                    'stage': stage,
                    'remaining_interventions': len(remaining_interventions) if not args.replacement else len(INTERVENTIONS),
                    'cumulative_timesteps': cumulative_timesteps
                })
            
            stage += 1
        
        # generate plots after training
        sb3_progress_path = aggregate_sb3_progress(args.log_dir)
        validation_csv_path = os.path.join(args.log_dir, 'validation_log.csv')
        if sb3_progress_path:
            visualizer.plot_training_and_validation_curves(sb3_progress_path, validation_csv_path)

        # final evaluation
        final_performance = evaluate_final_performance(student_model, args.task, num_episodes=20, seed=args.seed + 999)

        # summary
        logging.info("curriculum sequencing completed")
        logging.info("sequence order:")

        for i, result in enumerate(completed_sequence, 1):
            if result['test_metrics']:
                avg_reward = result['test_metrics'].get('avg_reward')
                cm_score = result['test_metrics'].get('cm_score')
                if avg_reward is not None:
                    logging.info(f"{i}. {(result['intervention'] if result['intervention'] is not None else 'none')} (test_reward: {avg_reward:.3f})")
                elif cm_score is not None:
                    logging.info(f"{i}. {(result['intervention'] if result['intervention'] is not None else 'none')} (cm_score: {cm_score:.3f})")
                else:
                    logging.info(f"{i}. {(result['intervention'] if result['intervention'] is not None else 'none')} (test_metric: None)")
            else:
                logging.info(f"{i}. {(result['intervention'] if result['intervention'] is not None else 'none')} (test_metric: None)")
        
        logging.info(f"\nperformance comparison")
        logging.info(f"initial: {initial_performance['avg_reward']:.3f} reward, {initial_performance['success_rate']:.3f} success")
        logging.info(f"final: {final_performance['avg_reward']:.3f} reward, {final_performance['success_rate']:.3f} success")
        logging.info(f"improvement: {final_performance['avg_reward'] - initial_performance['avg_reward']:.3f} reward")
        
        # Save final model
        final_model_path = os.path.join(args.log_dir, "final_model_after_sequencing.zip")
        student_model.save(final_model_path)
        logging.info(f"Final model saved to {final_model_path}")

        # save sequence results
        import json
        results_path = os.path.join(args.log_dir, "sequencing_results.json")
        with open(results_path, 'w') as f:
            json.dump({
                'sequence': completed_sequence,
                'initial_performance': initial_performance,
                'final_performance': final_performance,
                'cumulative_timesteps': cumulative_timesteps,
                'args': vars(args)
            }, f, indent=2)
        logging.info(f"results saved to {results_path}")

        if args.use_wandb:
            wandb.log({
                'final_reward': final_performance['avg_reward'],
                'final_success_rate': final_performance['success_rate'],
                'total_improvement': final_performance['avg_reward'] - initial_performance['avg_reward'],
                'total_timesteps': cumulative_timesteps
            })
            wandb.finish()
        
        # Generate enhanced comprehensive visualizations
        try:
            from src.visualization import create_enhanced_visualizations
            
            logging.info("🎨 Generating enhanced comprehensive visualizations...")
            # Use the logs directory as the base directory for finding all log directories
            logs_base_dir = 'logs'
            
            # Generate visualizations for all available heuristics (including replacement variants)
            create_enhanced_visualizations(
                log_base_dir=logs_base_dir,
                heuristics=['greedy', 'greedy_replacement', 'cm', 'cm_replacement', 'none', 'random', 'random_replacement', 'rnd', 'count'],
                output_dir=os.path.join(logs_base_dir, 'comprehensive_visualizations')
            )
            
            logging.info("✅ Enhanced visualizations completed!")
            
        except ImportError:
            logging.warning("Enhanced visualization module not available")
        except Exception as e:
            logging.warning(f"Enhanced visualization failed: {e}")
            logging.info("Continuing with basic visualizations only")
    
    if args.eval:
        gen_eval(args.log_dir, task_name=args.task, seed=args.seed)

def gen_eval(log_dir, task_name='pushing', seed=0, max_episode_length=250, skip_frame=3, num_episodes=10):
    set_seed(seed)
    print(f"running evaluation...")
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

    model_path = os.path.join(log_dir, 'final_model_after_sequencing.zip')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}")

    model = PPO.load(model_path)

    logging.info("\nFirst phase of evaluation:")
    all_rewards, all_successes = [], []
    for ep in range(num_episodes):
        obs = env.reset(seed=seed+ep)
        done = False
        total_reward, successes = 0.0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if isinstance(info, dict) and 'success' in info:
                successes += int(info['success'])
        logging.info(f"episode {ep + 1}: reward = {total_reward:.2f}, success = {successes}")
        all_rewards.append(total_reward)
        all_successes.append(successes)
    
    logging.info(f"\nmean reward: {np.mean(all_rewards):.2f}")
    logging.info(f"mean success rate: {np.mean(all_successes):.2f}")

    logging.info("\nGenerating benchmark evaluation and visualization...")
    if task_name not in TASK_BENCHMARKS:
        logging.info(f"No benchmark available for task: '{task_name};. Supported: {SUPPORTED_TASKS}")
        return
    benchmark = TASK_BENCHMARKS[task_name]

    # using the CausalWorld benchmark
    evaluation = EvaluationPipeline(
        evaluation_protocols=benchmark['evaluation_protocols'],
        task_params={'task_generator_id': task_name},
        world_params={'skip_frame': 3, 'action_mode': 'joint_torques'},
        # policy_class=PPO,
        # policy_path=model_path,
        visualize_evaluation=False
    )
    def policy_fn(obs):
        model = PPO.load(model_path)
        action, _ = model.predict(obs, deterministic=True)
        return action
    
    scores_model = evaluation.evaluate_policy(policy_fn, fraction=0.005)
    
    logging.info("\nEvaluation Results:")
    logging.info(scores_model)

    import json
    benchmark_path = os.path.join(log_dir, "benchmark_results.json")
    with open(benchmark_path, 'w') as f:
        json.dump({
            'final_evals': scores_model
        }, f, indent=2)
    logging.info(f"final evals saved to {benchmark_path}")
    
    plots_dir = os.path.join(log_dir, "plots")
    vis.generate_visual_analysis(plots_dir, experiments={task_name: scores_model})
    print("Visualization saved to:", plots_dir)

if __name__ == '__main__':
    main()
