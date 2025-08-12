import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import argparse
import logging
from collections import deque
from copy import deepcopy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from causal_world.task_generators import generate_task
from causal_world.envs import CausalWorld
from causal_world.intervention_actors import (
    GoalInterventionActorPolicy, 
    PhysicalPropertiesInterventionActorPolicy,
    VisualInterventionActorPolicy,
    RigidPoseInterventionActorPolicy,
    RandomInterventionActorPolicy
)
import wandb

# =====================
# Utility: Set random seed
# =====================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logging.info(f"Random seed set to {seed}")

# Ensure log directory exists and configure logging at the top (Python 3.7 compatible)
os.makedirs('meta_teacher_student_logs', exist_ok=True)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(process)d %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('meta_teacher_student_logs', 'meta_teacher_student.log')),
        logging.StreamHandler()
    ]
)

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
        # adding a hidden layer for better capacity
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
        # self.fc_mu_logvar = nn.Linear(input_dim, latent_dim * 2)
        # self.decoder = nn.Linear(latent_dim, input_dim)
        self.beta = beta
        # initialize weights properly
        self._init_weights()
    def _init_weights(self):
        """initialize weights to prevent initial instability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    def reparameterize(self, mu, logvar):
        # clamp logvar before exponentiation
        logvar = torch.clamp(logvar, min=-10, max=2)  # a conservative upper bound
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x):
        # normalize input if not done already
        x = torch.clamp(x, min=-10, max=10)
        mu_logvar = self.encoder(x)
        # mu_logvar = self.fc_mu_logvar(x)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

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

        print(f"IntervenedCausalWorld created with {intervention['type']} intervention")

    def reset(self):
        """
        Resets the environment and then applies the intervention.
        """
        self.reset_count += 1
        # logging.info(f"[ENV RESET] Resetting environment (reset_count={self.reset_count})")
        obs = self.base_env.reset()
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
                    print(f"Reset #{self.reset_count}: {self.intervention['type']} intervention applied (success: {success_signal})")
            else:
                print(f"Reset #{self.reset_count}: No intervention dict for {self.intervention['type']}")
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

# =====================
# Utility: Evaluate CM Score
# =====================
def evaluate_cm_score(env, student_model, episodes=5, device='cpu', intervention_type="unknown"):
    print(f"Evaluating CM score for {intervention_type} intervention...")
    # Collect data using student policy
    data = []
    total_steps = 0
    total_reward = 0
    success_count = 0
    termination_reasons = []
    for episode in range(episodes):
        obs = env.reset()
        done = False
        episode_steps = 0
        episode_reward = 0
        while not done:
            act, _ = student_model.predict(obs, deterministic=True)
            next_obs, rew, done, info = env.step(act)
            # log termination reason
            if done:
                if isinstance(info, dict) and 'success' in info and info['success']:
                    termination_reasons.append('success')
                    success_count += 1
                elif episode_steps >= 500:
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
    print(f"average episode length: {total_steps/episodes:.1f}")
    print(f"average episode reward: {total_reward/episodes:.3f}")
    print(f"termination reasons: {termination_reasons}")
    print(f"success rate: {success_count}/{episodes}")
    if len(data) == 0:
        print("no data collected! returning CM score of 0")
        return 0.0

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

def normalize_cm_scores(cm_scores):
    """normalize cm scores using z-score normalization to improve DQN stability"""
    mean = np.mean(cm_scores)
    std = np.std(cm_scores)
    if std < 1e-8:  # avoid division by zero
        print(f"warning: cm scores have very low variance (std={std:.6f})")
        return cm_scores
    normalized = (cm_scores - mean) / std
    return normalized.astype(np.float32)

# =====================
# Utility: Get Teacher State (CM scores for all interventions)
# =====================
def get_teacher_state(student_model, task_name, interventions, device='cpu', seed=0):
    print(f"Computing teacher state (CM scores for all interventions)...")
    cm_scores = []
    for i, intervention in enumerate(interventions):
        print(f"Processing intervention {i+1}/{len(interventions)}: {intervention['type']}")
        # create the base environment
        dense_weights = DENSE_REWARD_WEIGHTS.get(task_name, [0])
        task = generate_task(task_generator_id=task_name, dense_reward_weights=np.array(dense_weights), 
                            variables_space='space_a', fractional_reward_weight=1)
        base_env = CausalWorld(
            task=task,
            skip_frame=3,
            action_mode='joint_torques',
            enable_visualization=False,
            seed=seed,
            max_episode_length=5000
        )
        intervened_env = IntervenedCausalWorld(base_env, intervention)
        cm_score = evaluate_cm_score(intervened_env, student_model, device=device, intervention_type=intervention['type'])
        cm_scores.append(cm_score)
        base_env.close()
        print(f"{intervention['type']} is complete. CM score: {cm_score:.4f}")
    cm_array = np.array(cm_scores, dtype=np.float32)
    cm_normalized = normalize_cm_scores(cm_array)
    print(f"Raw CM scores: {cm_array}")
    print(f"Normalized CM scores: {cm_normalized}")
    return cm_normalized

# =====================
# Utility: Evaluate Student Performance (Success Rate)
# =====================
def evaluate_student_performance(student_model, validation_env, num_episodes=5):
    print(f"\nevaluating student performance ({num_episodes} episodes)...")
    successes = 0
    total_rewards = 0
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs = validation_env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        episode_success = False
        
        while not done:
            act, _ = student_model.predict(obs, deterministic=True)
            obs, reward, done, info = validation_env.step(act)
            episode_reward += reward
            episode_length += 1
            # logging.info(f"[EVAL STEP] Episode {episode+1}, Step {episode_length}, Reward: {reward}, Done: {done}, Info: {info}")
            if isinstance(info, dict) and 'success' in info:
                if info['success'] and not episode_success:
                    successes += 1
                    episode_success = True
                    done = True
                    break  # Early stop on success
        
        total_rewards += episode_reward
        episode_lengths.append(episode_length)
    
    success_rate = successes / num_episodes
    avg_reward = total_rewards / num_episodes
    avg_length = np.mean(episode_lengths)
    
    print(f"performance summary:")
    print(f"success rate: {success_rate:.3f} ({successes}/{num_episodes})")
    print(f"average reward: {avg_reward:.3f}")
    print(f"average episode length: {avg_length:.1f}")
    
    # logging.info(f"[EVAL SUMMARY] Success rate: {success_rate}, Avg reward: {avg_reward}, Avg length: {avg_length}")
    return success_rate, avg_reward, avg_length

# =====================
# adding a callback to monitor rewards
# =====================
class RewardMonitorCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.step_count = 0
        
    def _on_step(self) -> bool:
        self.step_count += 1
        
        # Check if we have infos and they contain episode data
        if (hasattr(self.locals, 'infos') and 
            self.locals['infos'] is not None and 
            len(self.locals['infos']) > 0):
            
            for info in self.locals['infos']:
                if isinstance(info, dict) and 'episode' in info:
                    episode_info = info['episode']
                    self.episode_rewards.append(episode_info['r'])
                    self.episode_lengths.append(episode_info['l'])
                    
                    # Check for success if available
                    if 'success' in info:
                        self.episode_successes.append(info['success'])
                    
                    # Log recent episode completion
                    if len(self.episode_rewards) <= 5:  # Log first 5 episodes
                        print(f"Episode completed: reward={episode_info['r']:.3f}, length={episode_info['l']}")
        
        return True
    
    def get_stats(self):
        """Get recent training statistics"""
        if not self.episode_rewards:
            return {"mean_reward": 0.0, "mean_length": 0.0, "episodes": 0, "success_rate": 0.0}
        
        recent_rewards = self.episode_rewards[-10:]  # Last 10 episodes
        recent_lengths = self.episode_lengths[-10:]
        
        return {
            "mean_reward": np.mean(recent_rewards),
            "mean_length": np.mean(recent_lengths),
            "episodes": len(self.episode_rewards),
            "success_rate": np.mean(self.episode_successes[-10:]) if self.episode_successes else 0.0
        }

class MetaLearningMonitor:
    def __init__(self, log_dir, interventions):
        self.log_dir = log_dir
        self.interventions = interventions
        self.q_value_history = []
        self.cm_score_history = []
        self.reward_history = []
        self.replay_buffer_rewards = []
    
    def log_q_values(self, teacher, state):
        """log the full q-vector after each update"""
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(teacher.device)
            q_values = teacher.q_net(state_tensor).cpu().numpy()[0]
        self.q_value_history.append(q_values.copy())
        print(f"Q-values: {[f'{q:.3f}' for q in q_values]}")
        # check for divergence
        if len(self.q_value_history) > 1:
            q_std = np.std(q_values)
            print(f"Q-value standard deviation: {q_std:.4f}")
            if q_std < 0.01:
                print(f"WARNING: Q-values are not diverging - may indicate learning issues")
        return q_values
    
    def log_cm_scores(self, cm_scores):
        """track cm score variance across interventions"""
        self.cm_score_history.append(cm_scores.copy())

        cm_variance = np.var(cm_scores)
        cm_range = np.max(cm_scores) - np.min(cm_scores)
        cm_std = np.std(cm_scores)

        # check for discriminatory power
        relative_std = cm_std / (np.mean(np.abs(cm_scores)) + 1e-8)
        if relative_std < 0.01:
            print(f"WARNING: CM scores lack discriminatory power (all values within 1%)")
        
        return {
            'variance': cm_variance,
            'range': cm_range,
            'std': cm_std,
            'relative_std': relative_std
        }


# =====================
# Convergence criteria
# =====================
class MetaLearningStagnationDetector:
    def __init__(self, patience_steps=50000, tolerance=1e-4, min_episodes=5):
        self.patience_steps = patience_steps
        self.tolerance = tolerance
        self.min_episodes = min_episodes

        # track the variables
        self.metrics_history = {
            'validation_success': [],
            'validation_reward': [],
            'teacher_reward': [],
            'student_success': []
        }
        self.last_significant_change = 0
        self.total_steps = 0
    
    def update(self, validation_success, validation_reward, teacher_reward, student_success):
        """update metrics and check for stagnation"""
        self.total_steps += 1

        # store current metrics
        current_metrics = {
            'validation_success': validation_success,
            'validation_reward': validation_reward,
            'teacher_reward': teacher_reward,
            'student_success': student_success
        }

        # add to history
        for key, value in current_metrics.items():
            self.metrics_history[key].append(value)
        
        # check for significant changes
        if self._has_significant_change():
            self.last_significant_change = self.total_steps
        
        return self.should_stop()
    
    def _has_significant_change(self):
        """check if any metric has changed significantly"""
        if len(self.metrics_history['validation_success']) < self.min_episodes:
            return True
        
        # look at recent window versus previous window
        window_size = min(5, len(self.metrics_history['validation_success']) // 2)

        for metric_name, values in self.metrics_history.items():
            if len(values) < window_size * 2:
                continue
            
            recent_window = values[-window_size:]
            previous_window = values[-window_size*2:-window_size]

            recent_mean = np.mean(recent_window)
            previous_mean = np.mean(previous_window)

            # check for significant change
            change = abs(recent_mean - previous_mean)
            if change > self.tolerance:
                return True
        
        print(f"no significant change detected in {metric_name}:{change:.6f}")
        return False
    
    def should_stop(self):
        """determine if training should stop due to stagnation"""
        steps_since_change = self.total_steps - self.last_significant_change
        
        if steps_since_change >= self.patience_steps:
            print(f"stopping due to stagnantion: {steps_since_change} steps without significant change")
            return True
        
        return False
    
    def get_status(self):
        """get current status for logging"""
        steps_since_change = self.total_steps - self.last_significant_change
        return {
            'steps_since_change': steps_since_change,
            'patience_remaining': max(0, self.patience_steps - steps_since_change),
            'total_episodes': len(self.metrics_history['validation_success'])
        }


# =====================
# Teacher DQN
# =====================
class ReplayBuffer:
    def __init__(self, capacity, state_dim, device):
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
    def push(self, s, a, r, s_):
        self.buffer.append((s, a, r, s_))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_ = zip(*batch)
        return (
            torch.tensor(np.stack(s), dtype=torch.float32).to(self.device),
            torch.tensor(a, dtype=torch.long).to(self.device),
            torch.tensor(r, dtype=torch.float32).to(self.device),
            torch.tensor(np.stack(s_), dtype=torch.float32).to(self.device)
        )
    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class TeacherDQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, device='cpu', buffer_size=1500, batch_size=6, target_update=5):
        self.device = device
        self.q_net = DQN(state_dim, action_dim).to(device)
        self.target_net = deepcopy(self.q_net)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.replay_buffer = ReplayBuffer(buffer_size, state_dim, device)
        self.steps = 0
        self.update_count = 0
    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.q_net.net[-1].out_features - 1)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state)
        return int(q_values.argmax().item())
    def train_step(self):
        if len(self.replay_buffer) >= self.batch_size:
            s, a, r, s_ = self.replay_buffer.sample(min(self.batch_size, len(self.replay_buffer)))
            q_values = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                max_next_q = self.target_net(s_).max(1)[0]
                target = r + self.gamma * max_next_q
            loss = nn.MSELoss()(q_values, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.steps += 1
            self.update_count += 1
            if self.steps % self.target_update == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())
            return loss.item()
        return None
        

# =====================
# Main Meta-Learning Loop
# =====================
def main():
    parser = argparse.ArgumentParser(description="Meta-RL Teacher-Student Curriculum for CausalWorld")
    parser.add_argument('--task', type=str, default='pushing', help='Task name')
    parser.add_argument('--student_train_steps', type=int, default=50000, help='Timesteps per student training block')
    parser.add_argument('--meta_episodes', type=int, default=30, help='Number of meta-episodes (teacher steps)')
    parser.add_argument('--teacher_lr', type=float, default=1e-3, help='Teacher DQN learning rate')
    parser.add_argument('--teacher_gamma', type=float, default=0.99, help='Teacher DQN discount factor')
    parser.add_argument('--teacher_buffer_size', type=int, default=1500, help='Teacher DQN replay buffer size')
    parser.add_argument('--teacher_batch_size', type=int, default=6, help='Teacher DQN batch size')
    parser.add_argument('--teacher_target_update', type=int, default=5, help='Teacher DQN target update frequency')
    parser.add_argument('--student_pretrained_path', type=str, default=None, help='Path to pretrained PPO model (optional)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--use_wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--log_dir', type=str, default='meta_teacher_student_logs', help='Log directory')
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)
    # Removed logging.basicConfig from here

    # Auto-determine pretrained path if not provided
    if args.student_pretrained_path is None:
        args.student_pretrained_path = f'ppo_{args.task}_sb3/final_model.zip'
        logging.info(f"🔍 Auto-determined pretrained path: {args.student_pretrained_path}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.use_wandb:
        wandb.init(
            project=f'meta-teacher-student-{args.task}',
            name=f'meta_teacher_student_{args.task}_seed{args.seed}',
            config=vars(args),
            tags=[args.task, 'MetaRL', 'TeacherStudent', 'DQN', 'PPO', 'curriculum'],
            sync_tensorboard=True
        )

    print(f"\n{'='*50}")
    print(f"starting meta-rl teacher-student training")
    print(f"   task: {args.task}")
    print(f"   meta-episodes: {args.meta_episodes}")
    print(f"   student training steps: {args.student_train_steps}")
    print(f"{'='*50}")

    # --- Student Agent (PPO) ---
    if args.student_pretrained_path and os.path.exists(args.student_pretrained_path):
        student_model = PPO.load(args.student_pretrained_path)
        logging.info(f"Loaded student PPO from {args.student_pretrained_path}")
    else:
        # Create a new PPO model
        dense_weights = DENSE_REWARD_WEIGHTS.get(args.task, [0])
        task = generate_task(
            task_generator_id=args.task,
            dense_reward_weights=np.array(dense_weights),
            variables_space='space_a',
            fractional_reward_weight=1
        )
        base_env = CausalWorld(
            task=task,
            skip_frame=3,
            action_mode='joint_torques',
            enable_visualization=False,
            seed=args.seed,
            max_episode_length=5000
        )
        env = DummyVecEnv([lambda: base_env])
        env = VecMonitor(env, filename=os.path.join(args.log_dir, 'monitor.csv'))
        student_model = PPO('MlpPolicy', env, verbose=0, seed=args.seed)
        logging.info("Initialized new student PPO model.")

    # --- Teacher Agent (DQN) ---
    state_dim = len(INTERVENTIONS)
    action_dim = len(INTERVENTIONS)
    teacher = TeacherDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=args.teacher_lr,
        gamma=args.teacher_gamma,
        device=device,
        buffer_size=args.teacher_buffer_size,
        batch_size=args.teacher_batch_size,
        target_update=args.teacher_target_update
    )

    # --- Validation Environment (no intervention) ---
    dense_weights = DENSE_REWARD_WEIGHTS.get(args.task, [0])
    task = generate_task(
        task_generator_id=args.task,
        dense_reward_weights=np.array(dense_weights),
        variables_space='space_a',
        fractional_reward_weight=1
    )
    validation_env = CausalWorld(
        task=task,
        skip_frame=3,
        action_mode='joint_torques',
        enable_visualization=False,
        seed=args.seed+42,
        max_episode_length=5000
    )

    # --- Environment Factory ---
    def create_env_factory(intervention, task_name, seed, meta_episode):
        def _make_env():
            dense_weights = DENSE_REWARD_WEIGHTS.get(task_name, [0])
            task = generate_task(
                task_generator_id=task_name,
                dense_reward_weights=np.array(dense_weights),
                variables_space='space_a',
                fractional_reward_weight=1
            )
            base_env = CausalWorld(
                task=task,
                skip_frame=3,
                action_mode='joint_torques',
                enable_visualization=False,
                seed=seed + meta_episode*100,  # adding some randomness
                max_episode_length=5000
            )
            env = DummyVecEnv([lambda: IntervenedCausalWorld(base_env, intervention)])
            env = VecMonitor(env, filename=os.path.join(args.log_dir, 'monitor.csv'))
            return env
        return _make_env
    
    def get_epsilon(meta_episode, meta_episodes, epsilon_start=1.0, epsilon_end=0.3):
        if meta_episode < meta_episodes:
            # linear decay from 1.0 to 0.3 over the specified number of meta-episodes (30 as of now)
            return epsilon_start - (epsilon_start - epsilon_end) * (meta_episode / meta_episodes)
        else:
            return epsilon_end
    
    # --- Initial Student Performance ---
    print("\n" + "="*50)
    print("initial evaluation")
    print("="*50)
    student_performance_before, avg_reward_before, avg_length_before = evaluate_student_performance(student_model, validation_env)
    print(f"initial student success rate: {student_performance_before:.3f}")

    # --- Initialize stagnantion detector ---
    stagnation_detector = MetaLearningStagnationDetector(
        patience_steps=50000,   # stop after 50k steps without change
        tolerance=0.01,         # 1% change threshold
        min_episodes=10         # need at least 10 episodes of data
    )
    monitor = MetaLearningMonitor(args.log_dir, INTERVENTIONS)

    # --- Meta-Learning Loop ---
    for meta_episode in range(args.meta_episodes):
        # reduced learning rate to prevent catastrophic forgetting
        if meta_episode > 0:
            student_model.learning_rate = max(1e-5, 3e-4 * (0.9 ** meta_episode))  # gradual
        # 1. Get Teacher State (CM scores)
        S_teacher = get_teacher_state(student_model, args.task, INTERVENTIONS, device=device, seed=args.seed)
        cm_stats = monitor.log_cm_scores(S_teacher)
        # 2. Teacher selects action
        epsilon = get_epsilon(meta_episode, args.meta_episodes)
        intervention_index = teacher.select_action(S_teacher, epsilon)
        chosen_intervention = INTERVENTIONS[intervention_index]
        # 3. Train Student
        env_factory = create_env_factory(chosen_intervention, args.task, args.seed, meta_episode)
        student_model.set_env(env_factory())
        callback = RewardMonitorCallback()
        student_model.learn(total_timesteps=args.student_train_steps, callback=callback, reset_num_timesteps=False)
        # after student training...
        save_path = os.path.join(args.log_dir, f"temp_student_model_episode_{meta_episode}.zip")
        student_model.save(save_path)
        print(f"saved student model to {save_path}")
        # 4. Calculate Teacher Reward
        student_performance_after, avg_reward_after, avg_length_after = evaluate_student_performance(student_model, validation_env)
        R_teacher = student_performance_after - student_performance_before
        monitor.reward_history.append(R_teacher)
        # 5. Get Next Teacher State
        S_prime_teacher = get_teacher_state(student_model, args.task, INTERVENTIONS, device=device, seed=args.seed)
        # 6. Update Teacher
        teacher.replay_buffer.push(S_teacher, intervention_index, R_teacher, S_prime_teacher)
        loss = teacher.train_step()
        # 7. Logging & Stagnation detection
        should_stop = stagnation_detector.update(
            validation_success=student_performance_after,
            validation_reward=avg_reward_after,
            teacher_reward=R_teacher,
            student_success=callback.get_stats()['success_rate']   # from training callback
        )
        # log the stagnation status
        status = stagnation_detector.get_status()
        logging.info(f"Meta-Episode {meta_episode+1}/{args.meta_episodes}: Teacher chose '{chosen_intervention['type']}', Reward: {R_teacher:.4f}, Student Success: {student_performance_after:.3f}, Loss: {loss:.6f if loss else 'N/A'}")
        logging.info(f"Meta-Episode {meta_episode+1}: Stagnation check - steps_since_change={status['steps_since_change']}, patience_remaining={status['patience_remaining']}")
        # track metrics...
        progress_metrics = {
            'validation_success': student_performance_after,
            'validation_reward': avg_reward_after,
            'avg_cm_score': np.mean(S_prime_teacher),
            'teacher_q_values': teacher.q_net(torch.tensor(S_prime_teacher, dtype=torch.float32).to(device)).detach().cpu().numpy().tolist()
        }
        if args.use_wandb:
            wandb.log({
                'meta_episode': meta_episode,
                'teacher_action': chosen_intervention['type'],
                'teacher_reward': R_teacher,
                'student_success_rate': student_performance_after,
                'epsilon': epsilon,
                'cm_scores': {INTERVENTIONS[i]['type']: float(S_teacher[i]) for i in range(len(INTERVENTIONS))},
                'stagnation_steps': status['steps_since_change'],
                'patience_remaining': status['patience_remaining']
            })
            wandb.log(progress_metrics)
        # early stopping check
        if should_stop:
            logging.info(f"Early stopping triggered at meta episode {meta_episode+1}")
            break
        # Update for next step
        student_performance_before = student_performance_after
        # before next iteration...
        if os.path.exists(save_path):
            print(f"loading student model from {save_path}")
            student_model = PPO.load(save_path)
        else:
            print(f"warning: no saved model found for episode {meta_episode}")
    if args.use_wandb:
        wandb.finish()
    logging.info("Meta-RL Teacher-Student curriculum training complete.")

if __name__ == '__main__':
    import time
    main() 