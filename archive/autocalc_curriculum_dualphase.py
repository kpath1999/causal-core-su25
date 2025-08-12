import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import argparse
import pickle
from collections import deque
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import time
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

# Configure logging at the top of the file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(process)d %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler("autocalc_training_debug.log"),
        logging.StreamHandler()
    ]
)

# Task configuration
DENSE_REWARD_WEIGHTS = {
    'pushing': [750, 250, 100],
    'picking': [250, 0, 125, 0, 750, 0, 0, 0.005],
    'reaching': [100000, 0, 0, 0],
    'pick_and_place': [750, 50, 250, 0, 0.005],
    'stacking2': [750, 250, 250, 125, 0.005],
}

class PolicyDivergenceCalculator:
    """Calculate KL divergence between two Gaussian policies."""
    
    @staticmethod
    def gaussian_kl_divergence(mu1: torch.Tensor, sigma1: torch.Tensor, 
                              mu2: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:
        """
        Calculate KL divergence between two multivariate Gaussian distributions.
        KL(N(mu1, sigma1) || N(mu2, sigma2))
        """
        # Ensure tensors are on the same device
        device = mu1.device
        mu2, sigma2 = mu2.to(device), sigma2.to(device)
        
        # Add small epsilon for numerical stability
        eps = 1e-8
        sigma1 = torch.clamp(sigma1, min=eps)
        sigma2 = torch.clamp(sigma2, min=eps)
        
        # Calculate KL divergence for diagonal covariance matrices
        log_ratio = torch.log(sigma2) - torch.log(sigma1)
        variance_ratio = (sigma1 / sigma2) ** 2
        mean_diff_sq = ((mu1 - mu2) / sigma2) ** 2
        
        kl_div = 0.5 * torch.sum(log_ratio + variance_ratio + mean_diff_sq - 1, dim=-1)
        return kl_div

    @staticmethod
    def compute_policy_divergence(policy_before: PPO, policy_after: PPO,
                                 reference_states: np.ndarray) -> float:
        """
        Compute average KL divergence between two policies over reference states.
        """
        device = next(policy_before.policy.parameters()).device
        reference_states_tensor = torch.FloatTensor(reference_states).to(device)
        
        kl_divergences = []
        
        with torch.no_grad():
            # Get policy distributions for all reference states
            for state in reference_states_tensor:
                state = state.unsqueeze(0)  # Add batch dimension
                
                # Get distributions from both policies
                dist_before = policy_before.policy.get_distribution(state)
                dist_after = policy_after.policy.get_distribution(state)
                
                # Extract parameters (assuming Gaussian distribution)
                mu_before = dist_before.distribution.mean.squeeze(0)
                sigma_before = dist_before.distribution.stddev.squeeze(0)
                mu_after = dist_after.distribution.mean.squeeze(0)
                sigma_after = dist_after.distribution.stddev.squeeze(0)
                
                # Calculate KL divergence
                kl_div = PolicyDivergenceCalculator.gaussian_kl_divergence(
                    mu_before, sigma_before, mu_after, sigma_after
                )
                kl_divergences.append(kl_div.item())
        
        return np.mean(kl_divergences)

class InterventionTester:
    """An intervention tester with minimal contamination."""

    def __init__(self,
                 mini_episode_length: int = 30,  # very short episodes
                 max_test_episodes: int = 3,     # very few episodes
                 early_termination_threshold: float = 0.7): # lower threshold
        
        self.mini_episode_length = mini_episode_length
        self.max_test_episodes = max_test_episodes
        self.early_termination_threshold = early_termination_threshold
        self.baseline_cache = None
    
    def run_intervention_tests(self, model: PPO, env, available_interventions: List[Dict]) -> Dict[str, Dict]:
        """Run minimal tests for each available intervention with relative ranking."""

        # quick baseline assessment (cached)
        if self.baseline_cache is None:
            self.baseline_cache = self._quick_baseline_assessment(model, env)
        
        baseline_metrics = self.baseline_cache

        # randomize the intervention testing order to eliminate ordering effects
        intervention_order = available_interventions.copy()
        random.shuffle(intervention_order)

        # test each intervention in randomized order
        test_results = {}
        intervention_metrics_list = []

        for i, intervention in enumerate(intervention_order):
            intervention_type = intervention['type']
            logging.info(f"Testing intervention {i+1}/{len(intervention_order)}: {intervention_type}")

            intervention_actor = self._create_intervention_actor(intervention, env)
            intervention_metrics = self._run_minimal_test(model, env, intervention_actor)

            test_results[intervention_type] = {
                'metrics': intervention_metrics,
                'episodes_used': intervention_metrics.get('episodes_used', 0),
                'test_order': i + 1     # recording the order for later analysis
            }

            intervention_metrics_list.append((intervention_type, intervention_metrics))
        
        # calculate relative struggle scores using statistical ranking
        struggle_scores = self._calculate_relative_struggle_scores(baseline_metrics, intervention_metrics_list)

        # update test results with struggle scores
        for intervention_type, struggle_score in struggle_scores.items():
            test_results[intervention_type]['struggle_score'] = struggle_score
            logging.info(f"Relative struggle score for {intervention_type}: {struggle_score:.4f}")
        
        wandb.log({
            'intervention_testing/results': test_results
        })
        
        return test_results
    
    def _calculate_relative_struggle_scores(self, baseline: Dict[str, float], 
                                          intervention_metrics_list: List[Tuple[str, Dict]]) -> Dict[str, float]:
        """
        calculate struggle scores using relative ranking approach.
        apples-to-apples comparison and reduces hyperparameter sensitivity.
        """
        
        # Calculate degradation ratios for each intervention
        degradation_scores = []
        
        for intervention_type, metrics in intervention_metrics_list:
            # Calculate relative performance ratios
            baseline_reward = baseline['avg_reward'] + 1e-6
            baseline_success = baseline['success_rate'] + 1e-6
            baseline_length = baseline['avg_length'] + 1e-6
            
            reward_ratio = metrics['avg_reward'] / baseline_reward
            success_ratio = metrics['success_rate'] / baseline_success
            length_ratio = metrics['avg_length'] / baseline_length
            
            # Combined degradation score (lower = more degradation)
            degradation_score = (
                reward_ratio * 0.4 + 
                success_ratio * 0.4 + 
                (1.0 / length_ratio) * 0.2
            )
            
            degradation_scores.append((intervention_type, degradation_score))
        
        # Rank interventions by degradation (most degraded = highest struggle)
        degradation_scores.sort(key=lambda x: x[1])  # Sort by degradation score
        
        # Convert to struggle scores using ranking
        struggle_scores = {}
        num_interventions = len(degradation_scores)
        
        for rank, (intervention_type, degradation_score) in enumerate(degradation_scores):
            # Use ranking-based score (0 to 1, where 1 = most struggle)
            # This eliminates hyperparameter sensitivity
            struggle_score = rank / (num_interventions - 1) if num_interventions > 1 else 0.5
            
            # Add small bonus for actual degradation magnitude
            # but keep ranking as primary factor
            magnitude_bonus = (1.0 - degradation_score) * 0.1
            struggle_score = min(1.0, struggle_score + magnitude_bonus)
            
            struggle_scores[intervention_type] = struggle_score
        
        # Add statistical confidence check
        self._validate_ranking_confidence(degradation_scores)
        
        return struggle_scores

    def _validate_ranking_confidence(self, degradation_scores: List[Tuple[str, float]]):
        """Validate that the ranking is statistically meaningful."""
        if len(degradation_scores) < 2:
            return
        
        # calculate the spread of degradation scores
        scores = [score for _, score in degradation_scores]
        score_range = max(scores) - min(scores)
        score_std = np.std(scores)

        # if scores are too close together, ranking might not be reliable
        if score_range < 0.1: # small range
            logging.warning("Warning: Intervention scores are very similar - ranking may not be reliable.")
        elif score_std < 0.05: # low variance
            logging.warning("Warning: Low variance in intervention scores - consider more test episodes.")
        
        # log ranking confidence
        logging.info(f"Ranking confidence: range={score_range:.4f}, std={score_std:.4f}")

        # show the ranking
        logging.info("Intervention ranking (most to least struggle):")
        for i, (intervention_type, score) in enumerate(degradation_scores):
            rank = len(degradation_scores) - i # reverse order for ranking
            logging.info(f"{rank}. {intervention_type}: {score:.3f}")
    
    def _quick_baseline_assessment(self, model: PPO, env) -> Dict[str, float]:
        """Quick baseline assessment using minimal episodes."""
        tracker = PerformanceTracker()

        # run just 2 episodes for the baseline
        for episode in range(2):
            obs = env.reset()
            episode_reward = 0.0
            episode_success = False
            episode_length = 0

            done = False
            while not done and episode_length < self.mini_episode_length:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                episode_length += 1

                if isinstance(info, dict) and 'success' in info:
                    episode_success = bool(info['success'])
                
            tracker.add_episode(episode_reward, episode_success, episode_length)
        
        return tracker.get_metrics()
    
    def _run_minimal_test(self, model: PPO, env, intervention_actor) -> Dict[str, float]:
        """Run minimal test with early termination."""
        tracker = PerformanceTracker()
        episodes_used = 0

        for episode in range(self.max_test_episodes):
            # ensure clean environment state for each episode
            obs = env.reset()

            # apply intervention
            try:
                variables_dict = env.get_variable_space_used()
                intervention_dict = intervention_actor._act(variables_dict)
                if intervention_dict:
                    env.do_intervention(intervention_dict)
            except Exception as e:
                logging.warning(f"Warning: Failed to apply intervention: {e}")
            
            # run the mini-episode
            episode_reward = 0.0
            episode_success = False
            episode_length = 0

            done = False
            while not done and episode_length < self.mini_episode_length:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                episode_length += 1

                if isinstance(info, dict) and 'success' in info:
                    episode_success = bool(info['success'])
            
            tracker.add_episode(episode_reward, episode_success, episode_length)
            episodes_used += 1

            # early termination if success rate is high enough
            current_metrics = tracker.get_metrics()
            if (current_metrics['success_rate'] > self.early_termination_threshold and episodes_used >= 2):
                logging.info(f"Early termination: success rate {current_metrics['success_rate']:.2f}")
                break
        
        # ensure clean state after testing
        env.reset()

        metrics = tracker.get_metrics()
        metrics['episodes_used'] = episodes_used
        return metrics
    
    def _create_intervention_actor(self, intervention: Dict, env):
        """Create and initialize intervention actor."""
        intervention_class = intervention['class']
        intervention_params = intervention['params']

        actor = intervention_class(**intervention_params)
        actor.initialize(env)
        return actor

class InterventionSelector:
    """ an intervention selector that relies on struggle scoring. """

    def __init__(self):
        self.available_interventions = [
            {'type': 'goal', 'class': GoalInterventionActorPolicy, 'params': {}},
            {'type': 'mass', 'class': PhysicalPropertiesInterventionActorPolicy, 'params': {'group': 'tool'}},
            {'type': 'friction', 'class': PhysicalPropertiesInterventionActorPolicy, 'params': {'group': 'stage'}},
            {'type': 'visual', 'class': VisualInterventionActorPolicy, 'params': {}},
            {'type': 'joints', 'class': JointsInterventionActorPolicy, 'params': {}},
            {'type': 'pose', 'class': RigidPoseInterventionActorPolicy, 'params': {'positions': True, 'orientations': True}},
        ]

        self.used_interventions = set()
        self.test_results_history = []

    def select_next_intervention(self, model: PPO, env, tester: InterventionTester) -> Optional[Dict]:
        """ select next intervention based on the struggle score. """

        # get available interventions
        available = [i for i in self.available_interventions if i['type'] not in self.used_interventions]

        if not available:
            if 'random' not in self.used_interventions:
                logging.info("All standard interventions used, switching to random intervention for the final phase.")
                self.used_interventions.add('random')
                return {
                    'type': 'random',
                    'class': RandomInterventionActorPolicy,
                    'params': {}
                }
            else:
                return None
        
        # run minimal tests
        test_results = tester.run_intervention_tests(model, env, available)
        self.test_results_history.append(test_results)

        # select intervention with the highest struggle score
        struggle_scores = {intervention_type: results['struggle_score']
                           for intervention_type, results in test_results.items()}
        
        best_intervention_type = max(struggle_scores, key=struggle_scores.get)
        best_intervention = next(i for i in available if i['type'] == best_intervention_type)

        # mark as used
        self.used_interventions.add(best_intervention_type)

        # log results
        logging.info(f"Selected intervention: {best_intervention_type}")
        logging.info(f"Struggle scores: {struggle_scores}")

        return best_intervention

class PerformanceTracker:
    """Tracks performance metrics during intervention testing."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset tracking for new test."""
        self.episode_rewards = []
        self.episode_successes = []
        self.episode_lengths = []
        self.action_entropies = []

    def add_episode(self, reward: float, success: bool, length: int, action_entropy: float = None):
        """Add episode metrics."""
        self.episode_rewards.append(reward)
        self.episode_successes.append(success)
        self.episode_lengths.append(length)
        if action_entropy is not None:
            self.action_entropies.append(action_entropy)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get aggregated performance metrics."""
        if not self.episode_rewards:
            return {
                'avg_reward': 0.0,
                'success_rate': 0.0,
                'avg_length': 0.0,
                'reward_std': 0.0,
                'avg_entropy': 0.0
            }
        
        return {
            'avg_reward': np.mean(self.episode_rewards),
            'success_rate': np.mean(self.episode_successes),
            'avg_length': np.mean(self.episode_lengths),
            'reward_std': np.std(self.episode_rewards),
            'avg_entropy': np.mean(self.action_entropies) if self.action_entropies else 0.0
        }


class AutoCaLCCurriculumManager:
    """
    Automated Causal Learning Curriculum Manager using intervention selection and policy divergence.
    Designed to ensure minimal contamination during intervention testing.
    """
    
    def __init__(self, 
                 divergence_threshold: float = 0.1,
                 min_training_steps: int = 50000,
                 max_training_steps: int = 200000,
                 reference_states_size: int = 1000,
                 patience: int = 3,
                 mini_episode_length: int = 50,
                 max_test_episodes: int = 5,
                 testing_frequency_multiplier: float = 0.1):    # reducing testing frequency
        """
        Args:
            divergence_threshold: KL divergence threshold to advance curriculum
            min_training_steps: Minimum steps before checking divergence
            max_training_steps: Maximum steps per intervention
            reference_states_size: Number of reference states for comparison
            patience: Number of checks below threshold before advancing
        """
        
        self.divergence_threshold = divergence_threshold
        self.min_training_steps = min_training_steps
        self.max_training_steps = max_training_steps
        self.reference_states_size = reference_states_size
        self.patience = patience
        self.testing_frequency_multiplier = testing_frequency_multiplier

        # our testing components
        self.intervention_tester = InterventionTester(
            mini_episode_length=mini_episode_length,
            max_test_episodes=max_test_episodes
        )
        self.intervention_selector = InterventionSelector()

        # current state
        self.current_intervention = None
        self.current_intervention_actor = None
        self.reference_states = None
        self.policy_before = None
        self.steps_in_current_intervention = 0
        self.divergence_history = []
        self.intervention_history = []
        self.patience_counter = 0

        # testing phase tracking
        self._in_testing_phase = False
        self.testing_complete = False
        self.last_testing_step = 0
        self.testing_interval = int(min_training_steps * testing_frequency_multiplier)
    
    def should_advance_curriculum(self, model: PPO, replay_buffer_states: np.ndarray) -> bool:
        """Determine if curriculum should advance based on policy divergence."""
        # Don't check until minimum training steps
        if self.steps_in_current_intervention < self.min_training_steps:
            return False
            
        # Force advance if maximum steps reached
        if self.steps_in_current_intervention >= self.max_training_steps:
            logging.info(f"Max steps reached for intervention {self.current_intervention_idx}")
            return True
            
        # Sample reference states if not set
        if self.reference_states is None or self.policy_before is None:
            self.reference_states = self._sample_reference_states(replay_buffer_states)
            # Save policy snapshot before intervention
            self.policy_before = self._copy_policy(model)
            return False
            
        # Calculate policy divergence
        divergence = PolicyDivergenceCalculator.compute_policy_divergence(
            self.policy_before, model, self.reference_states
        )
        
        self.divergence_history.append({
            'intervention': self.current_intervention['type'] if self.current_intervention else 'none',
            'step': self.steps_in_current_intervention,
            'divergence': divergence
        })
        
        logging.info(f"Intervention: {self.current_intervention['type'] if self.current_intervention else 'none'}, "
              f"Step: {self.steps_in_current_intervention}: KL divergence = {divergence:.4f}")

        # Check if divergence is below threshold
        if divergence < self.divergence_threshold:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                logging.info(f"Advancing curriculum: divergence {divergence:.4f} < {self.divergence_threshold} "
                      f"for {self.patience} consecutive checks")
                return True
        else:
            self.patience_counter = 0
            
        return False
    
    def advance_intervention(self, model: PPO, replay_buffer_states: np.ndarray, env):
        """Advance to the next intervention using our selection method"""
        # Record intervention completion
        if self.current_intervention:
            self.intervention_history.append({
                'intervention_type': self.current_intervention['type'],
                'steps_trained': self.steps_in_current_intervention,
                'final_divergence': self.divergence_history[-1]['divergence'] if self.divergence_history else None
            })
        
        # enter testing phase with minimal contamination
        self._in_testing_phase = True
        start_time = time.time()

        # select next intervention using struggle scoring
        next_intervention = self.intervention_selector.select_next_intervention(
            model, env, self.intervention_tester
        )

        testing_duration = time.time() - start_time
        logging.info(f"Testing phase completed in {testing_duration:.2f} seconds")
        self._in_testing_phase = False

        if next_intervention is None:
            logging.info("All interventions completed. Curriculum finished.")
            self.current_intervention = None
            self.testing_complete = True
            return

        # create intervention actor
        intervention_class = next_intervention['class']
        intervention_params = next_intervention['params']
        self.current_intervention_actor = intervention_class(**intervention_params)
        self.current_intervention_actor.initialize(env)

        # update state
        self.current_intervention = next_intervention
        self.steps_in_current_intervention = 0
        self.patience_counter = 0

        # sample new reference states and save policy snapshot
        self.reference_states = self._sample_reference_states(replay_buffer_states)
        self.policy_before = self._copy_policy(model)

        logging.info(f"Advanced to intervention: {next_intervention['type']}")
        
        wandb.log({
            'curriculum/advance_intervention': {
                'selected_intervention': next_intervention['type'] if next_intervention else 'none',
                'intervention_history': self.intervention_history,
                'divergence_history': self.divergence_history,
                'used_interventions': list(self.intervention_selector.used_interventions),
                'test_results_history': self.intervention_selector.test_results_history
            }
        })
    
    def _sample_reference_states(self, replay_buffer_states: np.ndarray) -> np.ndarray:
        """Sample reference states from replay buffer."""
        if len(replay_buffer_states) < self.reference_states_size:
            return replay_buffer_states.copy()
        
        indices = np.random.choice(len(replay_buffer_states), 
                                 size=self.reference_states_size, 
                                 replace=False)
        return replay_buffer_states[indices].copy()
    
    def _copy_policy(self, model: PPO) -> PPO:
        """Create a deep copy of the policy for comparison."""
        temp_path = "/tmp/temp_policy_test_autocalc.zip"
        model.save(temp_path)
        policy_copy = PPO.load(temp_path)
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return policy_copy
    
    def get_current_intervention(self) -> Optional[Dict]:
        """Get current intervention configuration."""
        return self.current_intervention
    
    def is_curriculum_complete(self) -> bool:
        """Check if all interventions have been completed."""
        return self.testing_complete
    
    def step(self):
        """Increment step counter."""
        if not self._in_testing_phase: # not counting steps during testing
            self.steps_in_current_intervention += 1
    
    def save_progress(self, filepath: str):
        """Save curriculum progress."""
        progress = {
            'current_intervention': self.current_intervention,
            'steps_in_current_intervention': self.steps_in_current_intervention,
            'divergence_history': self.divergence_history,
            'intervention_history': self.intervention_history,
            'reference_states': self.reference_states,
            'used_interventions': list(self.intervention_selector.used_interventions),
            'test_results_history': self.intervention_selector.test_results_history,
            'testing_complete': self.testing_complete
        }
        with open(filepath, 'wb') as f:
            pickle.dump(progress, f)

class AutoCaLCEnvironment:
    """
    Environment wrapper for the AutoCaLC curriculum. Minimal contamination.
    """
    
    def __init__(self, base_env: CausalWorld, curriculum_manager: AutoCaLCCurriculumManager):
        self.base_env = base_env
        self.curriculum_manager = curriculum_manager
        self.episode_count = 0
        self.last_intervention_log = 0
        
    def reset(self):
        """Reset environment and apply current intervention."""
        obs = self.base_env.reset()
        self.episode_count += 1
        
        # Apply current intervention if active and not in testing phase
        if (not self.curriculum_manager._in_testing_phase and
            self.curriculum_manager.current_intervention_actor is not None and
            self.curriculum_manager.current_intervention is not None):

            try:
                logging.info("\n[AutoCaLC] Attempting intervention:")
                variables_dict = self.base_env.get_variable_space_used()
                intervention_dict = self.curriculum_manager.current_intervention_actor._act(variables_dict)
                if hasattr(self.base_env, 'get_variable_space_used'):
                    logging.info(f"Current variable space: {self.base_env.get_variable_space_used()}")
                if hasattr(self.base_env, 'get_intervention_space_a'):
                    logging.info(f"Intervention space A: {self.base_env.get_intervention_space_a()}")
                if hasattr(self.base_env, 'get_intervention_space_b'):
                    logging.info(f"Intervention space B: {self.base_env.get_intervention_space_b()}")
                if hasattr(self.base_env, 'get_intervention_space_a_b'):
                    logging.info(f"Intervention space A_B: {self.base_env.get_intervention_space_a_b()}")
                success_signal, obs = self.base_env.do_intervention(intervention_dict)
                logging.info(f"do_intervention success: {success_signal}")

                # reduced logging frequency to minimize overhead
                if self.episode_count - self.last_intervention_log >= 200:
                    logging.info(f"Applied {self.curriculum_manager.current_intervention['type']} intervention (episode {self.episode_count})")
                    self.last_intervention_log = self.episode_count
            
            except Exception as e:
                logging.warning(f"[AutoCaLC] Exception during do_intervention: {e}")
        
        return obs
    
    def step(self, action):
        """Step the environment."""
        return self.base_env.step(action)
    
    def advance_curriculum_if_needed(self, model: PPO, replay_buffer_states: np.ndarray):
        """Check and advance curriculum if conditions are met."""
        if self.curriculum_manager.should_advance_curriculum(model, replay_buffer_states):
            self.curriculum_manager.advance_intervention(model, replay_buffer_states, self.base_env)
            return True
        return False
     
    def __getattr__(self, name):
        """Delegate attribute access to base environment."""
        return getattr(self.base_env, name)

class AutoCaLCCallback(BaseCallback):
    """Callback to manage AutoCaLC curriculum during training. Minimal contamination."""
    
    def __init__(self, curriculum_manager: AutoCaLCCurriculumManager, 
                 check_freq: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.curriculum_manager = curriculum_manager
        self.check_freq = check_freq
        self.replay_buffer_states = deque(maxlen=10000)
        self.last_curriculum_check = 0
        
    def _on_step(self) -> bool:
        # collect states for replay buffer (only during training, not testing)
        if not self.curriculum_manager._in_testing_phase:
            if hasattr(self.locals, 'obs_tensor'):
                obs = self.locals['obs_tensor'].cpu().numpy()
                self.replay_buffer_states.extend(obs)

            # increment curriculum step counter
            self.curriculum_manager.step()

        #  adaptive curriculum checking to minimize overhead
        current_step = self.num_timesteps
        if (current_step - self.last_curriculum_check >= self.check_freq and
            len(self.replay_buffer_states) > 0 and 
            not self.curriculum_manager.is_curriculum_complete() and 
            not self.curriculum_manager._in_testing_phase):

            self.last_curriculum_check = current_step
            replay_states = np.array(list(self.replay_buffer_states))

            # check if we should advance the curriculum
            if hasattr(self.training_env, 'envs'):
                env = self.training_env.envs[0]
                if hasattr(env, 'advance_curriculum_if_needed'):
                    advanced = env.advance_curriculum_if_needed(self.model, replay_states)
                    if advanced:
                        # log curriculum advancement
                        current_intervention = self.curriculum_manager.current_intervention
                        wandb.log({
                            'curriculum/intervention_type': current_intervention['type'] if current_intervention else 'none',
                            'curriculum/advancement_step': self.num_timesteps,
                            'curriculum/total_interventions_used': len(self.curriculum_manager.intervention_selector.used_interventions),
                            'curriculum/in_testing_phase': self.curriculum_manager._in_testing_phase,
                            'curriculum/testing_contamination_ratio': self._calculate_contamination_ratio(),
                            'curriculum/patience_counter': self.curriculum_manager.patience_counter,
                            'curriculum/steps_in_current_intervention': self.curriculum_manager.steps_in_current_intervention,
                            'curriculum/intervention_history': self.curriculum_manager.intervention_history,
                            'curriculum/divergence_history': self.curriculum_manager.divergence_history,
                            'curriculum/reference_states_size': len(self.curriculum_manager.reference_states) if self.curriculum_manager.reference_states is not None else 0,
                            'curriculum/testing_complete': self.curriculum_manager.testing_complete
                        }, step=self.num_timesteps)
                        # Log all test results for this advancement
                        if self.curriculum_manager.intervention_selector.test_results_history:
                            wandb.log({
                                'curriculum/last_test_results': self.curriculum_manager.intervention_selector.test_results_history[-1]
                            }, step=self.num_timesteps)
            
            # log metrics but only when in testing phase to avoid contamination
            if (self.curriculum_manager.divergence_history and not self.curriculum_manager.in_testing_phase):
                latest_divergence = self.curriculum_manager.divergence_history[-1]
                current_intervention = self.curriculum_manager.current_intervention

                wandb.log({
                    'curriculum/kl_divergence': latest_divergence['divergence'],
                    'curriculum/current_intervention': current_intervention['type'] if current_intervention else 'none',
                    'curriculum/intervention_steps': self.curriculum_manager.steps_in_current_intervention,
                    'curriculum/curriculum_complete': self.curriculum_manager.is_curriculum_complete(),
                    'curriculum/in_testing_phase': self.curriculum_manager._in_testing_phase,
                    'curriculum/testing_contamination_ratio': self._calculate_contamination_ratio(),
                    'curriculum/patience_counter': self.curriculum_manager.patience_counter,
                    'curriculum/intervention_history': self.curriculum_manager.intervention_history,
                    'curriculum/divergence_history': self.curriculum_manager.divergence_history,
                    'curriculum/reference_states_size': len(self.curriculum_manager.reference_states) if self.curriculum_manager.reference_states is not None else 0,
                    'curriculum/testing_complete': self.curriculum_manager.testing_complete
                }, step=self.num_timesteps)
        
        return True

    def _calculate_contamination_ratio(self) -> float:
        """Calculate the ratio of testing time to total training time."""
        if not self.curriculum_manager.intervention_history:
            return 0.0
        
        total_testing_steps = sum(
            self.curriculum_manager.intervention_tester.max_test_episodes *
            self.curriculum_manager.intervention_tester.mini_episode_length
            for _ in self.curriculum_manager.intervention_selector.test_results_history
        )

        total_training_steps = sum(
            history['steps_trained'] for history in self.curriculum_manager.intervention_history
        )

        if total_training_steps == 0:
            return 0.0

        return total_testing_steps / (total_training_steps + total_testing_steps)


def make_autocalc_env(rank: int, task_name: str, curriculum_manager: AutoCaLCCurriculumManager,
                      seed: int = 0, skip_frame: int = 3, max_episode_length: int = 250):
    """Create environment with AutoCaLC curriculum."""
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
        
        # Wrap with AutoCaLC curriculum
        env = AutoCaLCEnvironment(base_env, curriculum_manager)
        return env
    return _init

def train_autocalc_curriculum(
    num_envs: int,
    log_dir: str,
    max_episode_length: int,
    skip_frame: int,
    seed: int,
    ppo_config: Dict,
    total_timesteps: int,
    eval_freq: int,
    task_name: str,
    pretrained_path: str,
    curriculum_config: Dict,
    wandb_config: Optional[Dict] = None
):
    """Train PPO with AutoCaLC curriculum."""
    
    if wandb_config:
        wandb.init(
            project=wandb_config['project'],
            name=wandb_config['run_name'],
            config={
                'task_name': task_name,
                'curriculum_type': 'AutoCaLC_DualFramework',
                'max_episode_length': max_episode_length,
                'skip_frame': skip_frame,
                'seed': seed,
                'total_timesteps': total_timesteps,
                **ppo_config,
                **curriculum_config
            },
            tags=[task_name, 'PPO', 'AutoCaLC', 'curriculum', 'dual_phase', 'minimal_contamination'],
            sync_tensorboard=True
        )
    
    if not log_dir.startswith('autocalcdual'):
        log_dir = f'autocalcdual_{log_dir}'
    os.makedirs(log_dir, exist_ok=True)
    
    # Create curriculum manager
    curriculum_manager = AutoCaLCCurriculumManager(
        divergence_threshold=curriculum_config.get('divergence_threshold', 0.1),
        min_training_steps=curriculum_config.get('min_training_steps', 50000),
        max_training_steps=curriculum_config.get('max_training_steps', 200000),
        reference_states_size=curriculum_config.get('reference_states_size', 1000),
        patience=curriculum_config.get('patience', 3),
        mini_episode_length=curriculum_config.get('mini_episode_length', 50),
        max_test_episodes=curriculum_config.get('max_test_episodes', 5),
        testing_frequency_multiplier=curriculum_config.get('testing_frequency_multiplier', 0.1)
    )
    
    # Create vectorized environment
    env = SubprocVecEnv([
        make_autocalc_env(i, task_name, curriculum_manager, seed, skip_frame, max_episode_length)
        for i in range(num_envs)
    ])
    env = VecMonitor(env, filename=os.path.join(log_dir, 'monitor.csv'))
    
    # Create evaluation environment (without curriculum for consistent evaluation)
    def make_eval_env():
        dense_weights = DENSE_REWARD_WEIGHTS.get(task_name, [0])
        task = generate_task(
            task_generator_id=task_name,
            dense_reward_weights=np.array(dense_weights),
            variables_space='space_a',
            fractional_reward_weight=1
        )
        return CausalWorld(
            task=task,
            skip_frame=skip_frame,
            action_mode='joint_torques',
            enable_visualization=False,
            seed=seed,
            max_episode_length=max_episode_length
        )
    
    eval_env = make_eval_env()
    
    # Load pretrained model
    policy_kwargs = dict(activation_fn=nn.LeakyReLU, net_arch=[512, 256])
    model = PPO.load(
        pretrained_path,
        env=env,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        seed=seed,
        **ppo_config
    )
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(eval_freq // num_envs, 1),
        save_path=os.path.join(log_dir, 'logs'),
        name_prefix='autocalc_model_dualphase'
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
    
    autocalc_callback = AutoCaLCCallback(
        curriculum_manager=curriculum_manager,
        check_freq=10000,
        verbose=1
    )
    
    callbacks = [checkpoint_callback, eval_callback, autocalc_callback]
    
    if wandb_config:
        wandb_callback = WandbCallback(
            gradient_save_freq=10000,
            model_save_path=os.path.join(log_dir, 'wandb_models'),
            verbose=2
        )
        callbacks.append(wandb_callback)
    
    callback = CallbackList(callbacks)
    
    # Train model
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        tb_log_name="autocalc_curriculum_dualphase"
    )
    
    # Save final model and curriculum progress
    model.save(os.path.join(log_dir, 'final_autocalc_model_dualphase'))
    curriculum_manager.save_progress(os.path.join(log_dir, 'curriculum_progress_dualphase.pkl'))

    # Generate curriculum analysis plots
    plot_test_based_curriculum_analysis(curriculum_manager, log_dir)
    
    # Generate contamination analysis report
    contamination_report = generate_contamination_analysis(curriculum_manager, log_dir)
    logging.info(f"Contamination Analysis Report:")
    logging.info(contamination_report)
    
    if wandb_config:
        # Log final contamination metrics and full report
        wandb.log({
            'final/contamination_report': contamination_report,
            'final/contamination_ratio': contamination_report['contamination_ratio'],
            'final/total_testing_steps': contamination_report['total_testing_steps'],
            'final/total_training_steps': contamination_report['total_training_steps'],
            'final/testing_efficiency': contamination_report['testing_efficiency']
        })
        wandb.finish()
    
    logging.info(f"AutoCaLC training completed. Model saved to {log_dir}")
    logging.info(f"Contamination ratio: {contamination_report['contamination_ratio']:.4f}")
    logging.info(f"Testing efficiency: {contamination_report['testing_efficiency']:.2f} episodes per intervention")

def plot_test_based_curriculum_analysis(curriculum_manager: AutoCaLCCurriculumManager, save_dir: str):
    """Generate analysis plots for test-based curriculum learning."""
    
    plt.figure(figsize=(16, 12))
    
    # Plot 1: KL Divergence over time
    plt.subplot(2, 3, 1)
    if curriculum_manager.divergence_history:
        steps = [d['step'] for d in curriculum_manager.divergence_history]
        divergences = [d['divergence'] for d in curriculum_manager.divergence_history]
        
        plt.plot(steps, divergences, 'b-', alpha=0.7)
        plt.axhline(y=curriculum_manager.divergence_threshold, color='r', linestyle='--', 
                   label=f'Threshold ({curriculum_manager.divergence_threshold})')
        
        plt.xlabel('Training Steps')
        plt.ylabel('KL Divergence')
        plt.title('Policy Divergence During Test-Based Curriculum')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot 2: Intervention selection timeline
    plt.subplot(2, 3, 2)
    if curriculum_manager.intervention_history:
        intervention_types = [h['intervention_type'] for h in curriculum_manager.intervention_history]
        steps_trained = [h['steps_trained'] for h in curriculum_manager.intervention_history]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(intervention_types)))
        bars = plt.barh(range(len(intervention_types)), steps_trained, color=colors)
        plt.yticks(range(len(intervention_types)), intervention_types)
        plt.xlabel('Training Steps')
        plt.title('Time Spent per Selected Intervention')
        plt.grid(True, alpha=0.3)
        
        for i, (bar, steps) in enumerate(zip(bars, steps_trained)):
            plt.text(steps/2, i, f'{steps}', ha='center', va='center', fontweight='bold')
    
    # Plot 3: Test results for latest intervention selection
    plt.subplot(2, 3, 3)
    if curriculum_manager.intervention_selector.test_results_history:
        latest_tests = curriculum_manager.intervention_selector.test_results_history[-1]
        
        interventions = list(latest_tests.keys())
        struggle_scores = [latest_tests[i]['struggle_score'] for i in interventions]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(interventions)))
        bars = plt.bar(range(len(interventions)), struggle_scores, color=colors)
        plt.xticks(range(len(interventions)), interventions, rotation=45)
        plt.ylabel('Struggle Score')
        plt.title('Latest Intervention Test Results')
        plt.grid(True, alpha=0.3)
        
        # Highlight selected intervention
        max_idx = np.argmax(struggle_scores)
        bars[max_idx].set_edgecolor('red')
        bars[max_idx].set_linewidth(3)
    
    # Plot 4: Performance drops across all tests (commented out, as 'performance_drop' is not set)
    plt.subplot(2, 3, 4)
    # if curriculum_manager.intervention_selector.test_results_history:
    #     all_drops = {}
    #     for test_round in curriculum_manager.intervention_selector.test_results_history:
    #         for intervention, results in test_round.items():
    #             if intervention not in all_drops:
    #                 all_drops[intervention] = []
    #             all_drops[intervention].append(results['performance_drop'])
    #     interventions = list(all_drops.keys())
    #     avg_drops = [np.mean(all_drops[i]) for i in interventions]
    #     plt.bar(range(len(interventions)), avg_drops)
    #     plt.xticks(range(len(interventions)), interventions, rotation=45)
    #     plt.ylabel('Average Performance Drop')
    #     plt.title('Average Performance Impact by Intervention')
    #     plt.grid(True, alpha=0.3)
    plt.axis('off')
    
    # Plot 5: Selection order
    plt.subplot(2, 3, 5)
    if curriculum_manager.intervention_history:
        intervention_types = [h['intervention_type'] for h in curriculum_manager.intervention_history]
        order = list(range(1, len(intervention_types) + 1))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(intervention_types)))
        bars = plt.bar(order, [1]*len(intervention_types), color=colors)
        plt.xticks(order, [f'{i+1}' for i in range(len(intervention_types))])
        plt.ylabel('Intervention')
        plt.xlabel('Selection Order')
        plt.title('Test-Based Intervention Selection Sequence')
        
        for i, (bar, intervention) in enumerate(zip(bars, intervention_types)):
            plt.text(i+1, 0.5, intervention, ha='center', va='center', 
                    fontweight='bold', rotation=45, fontsize=8)
    
    # Plot 6: Testing efficiency
    plt.subplot(2, 3, 6)
    if curriculum_manager.intervention_selector.test_results_history:
        test_rounds = range(1, len(curriculum_manager.intervention_selector.test_results_history) + 1)
        interventions_tested = [len(test_round) for test_round in curriculum_manager.intervention_selector.test_results_history]
        
        plt.plot(test_rounds, interventions_tested, 'o-')
        plt.xlabel('Testing Round')
        plt.ylabel('Number of Interventions Tested')
        plt.title('Testing Efficiency Over Time')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'test_based_autocalc_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"📊 Enhanced test-based curriculum analysis plots saved to {save_dir}")

def generate_contamination_analysis(curriculum_manager: AutoCaLCCurriculumManager, save_dir: str) -> Dict:
    """Generate comprehensive contamination analysis report."""
    
    # Calculate contamination metrics
    total_testing_steps = 0
    total_testing_episodes = 0
    
    for test_round in curriculum_manager.intervention_selector.test_results_history:
        for intervention_type, results in test_round.items():
            episodes_used = results.get('episodes_used', curriculum_manager.intervention_tester.max_test_episodes)
            total_testing_episodes += episodes_used
            total_testing_steps += episodes_used * curriculum_manager.intervention_tester.mini_episode_length
    
    total_training_steps = sum(
        history['steps_trained'] for history in curriculum_manager.intervention_history
    )
    
    contamination_ratio = total_testing_steps / (total_training_steps + total_testing_steps) if (total_training_steps + total_testing_steps) > 0 else 0.0
    
    # Calculate testing efficiency
    total_interventions_tested = sum(len(test_round) for test_round in curriculum_manager.intervention_selector.test_results_history)
    testing_efficiency = total_testing_episodes / total_interventions_tested if total_interventions_tested > 0 else 0.0
    
    # Calculate early termination effectiveness
    early_terminations = 0
    total_possible_episodes = 0
    
    for test_round in curriculum_manager.intervention_selector.test_results_history:
        for intervention_type, results in test_round.items():
            episodes_used = results.get('episodes_used', curriculum_manager.intervention_tester.max_test_episodes)
            max_episodes = curriculum_manager.intervention_tester.max_test_episodes
            
            if episodes_used < max_episodes:
                early_terminations += 1
            
            total_possible_episodes += max_episodes
    
    early_termination_rate = early_terminations / total_interventions_tested if total_interventions_tested > 0 else 0.0
    
    # Generate detailed report
    report = {
        'contamination_ratio': contamination_ratio,
        'total_testing_steps': total_testing_steps,
        'total_training_steps': total_training_steps,
        'testing_efficiency': testing_efficiency,
        'early_termination_rate': early_termination_rate,
        'total_interventions_tested': total_interventions_tested,
        'total_testing_episodes': total_testing_episodes,
        'mini_episode_length': curriculum_manager.intervention_tester.mini_episode_length,
        'max_test_episodes': curriculum_manager.intervention_tester.max_test_episodes
    }
    
    # Save detailed report
    report_path = os.path.join(save_dir, 'contamination_analysis.json')
    with open(report_path, 'w') as f:
        import json
        json.dump(report, f, indent=2)
    
    return report

def main():
    parser = argparse.ArgumentParser(description='Test-Based AutoCaLC Curriculum Learning for CausalWorld')
    parser.add_argument('--train', action='store_true', help='Train with test-based AutoCaLC curriculum')
    parser.add_argument('--task', type=str, default='pushing', 
                       choices=list(DENSE_REWARD_WEIGHTS.keys()), help='Task name')
    parser.add_argument('--pretrained_path', type=str, 
                       help='Path to pretrained PPO model')
    parser.add_argument('--log_dir', type=str, default='autocalcdual_curriculum', help='Log directory')
    parser.add_argument('--timesteps', type=int, default=3_000_000, help='Total timesteps')
    parser.add_argument('--num_envs', type=int, default=16, help='Number of parallel environments')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--skip_frame', type=int, default=3, help='Frame skip')
    parser.add_argument('--max_episode_length', type=int, default=250, help='Max episode length')
    parser.add_argument('--eval_freq', type=int, default=100_000, help='Evaluation frequency')
    parser.add_argument('--use_wandb', action='store_true', help='Enable wandb logging')
    
    # Enhanced test-based AutoCaLC specific parameters
    parser.add_argument('--divergence_threshold', type=float, default=0.1, 
                       help='KL divergence threshold for curriculum advancement')
    parser.add_argument('--min_training_steps', type=int, default=50000, 
                       help='Minimum steps before checking divergence')
    parser.add_argument('--max_training_steps', type=int, default=200000, 
                       help='Maximum steps per intervention')
    parser.add_argument('--reference_states_size', type=int, default=1000, 
                       help='Number of reference states for comparison')
    parser.add_argument('--patience', type=int, default=3, 
                       help='Patience for curriculum advancement')
    parser.add_argument('--mini_episode_length', type=int, default=50,
                       help='Length of mini-episodes for testing (reduces contamination)')
    parser.add_argument('--max_test_episodes', type=int, default=5,
                       help='Maximum number of test episodes per intervention (reduces contamination)')
    parser.add_argument('--testing_frequency_multiplier', type=float, default=0.1,
                       help='Multiplier for testing frequency (lower = less contamination)')
    parser.add_argument('--early_termination_threshold', type=float, default=0.7,
                       help='Success rate threshold for early termination of tests')
    
    args = parser.parse_args()
    
    # Auto-determine pretrained path if not provided
    if args.pretrained_path is None:
        args.pretrained_path = f'ppo_{args.task}_sb3/final_model.zip'
        logging.info(f"🔍 Auto-determined pretrained path: {args.pretrained_path}")
    
    # PPO configuration
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
    
    # Enhanced test-based curriculum configuration
    curriculum_config = {
        'divergence_threshold': args.divergence_threshold,
        'min_training_steps': args.min_training_steps,
        'max_training_steps': args.max_training_steps,
        'reference_states_size': args.reference_states_size,
        'patience': args.patience,
        'mini_episode_length': args.mini_episode_length,
        'max_test_episodes': args.max_test_episodes,
        'testing_frequency_multiplier': args.testing_frequency_multiplier,
        'early_termination_threshold': args.early_termination_threshold
    }
    
    wandb_config = None
    if args.use_wandb:
        wandb_config = {
            'project': f'enhanced-test-based-autocalc-curriculum-{args.task}',
            'run_name': f'enhanced_autocalc_{args.task}_mini{args.mini_episode_length}_max{args.max_test_episodes}_freq{args.testing_frequency_multiplier}_seed{args.seed}'
        }
    
    if args.train:
        train_autocalc_curriculum(
            num_envs=args.num_envs,
            log_dir=args.log_dir,
            max_episode_length=args.max_episode_length,
            skip_frame=args.skip_frame,
            seed=args.seed,
            ppo_config=ppo_config,
            total_timesteps=args.timesteps,
            eval_freq=args.eval_freq,
            task_name=args.task,
            pretrained_path=args.pretrained_path,
            curriculum_config=curriculum_config,
            wandb_config=wandb_config
        )

if __name__ == '__main__':
    main()
