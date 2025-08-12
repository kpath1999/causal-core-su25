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
    VisualInterventionActorPolicy
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

class AutoCaLCCurriculumManager:
    """
    Automated Causal Learning Curriculum Manager using policy divergence.
    """
    
    def __init__(self, 
                 interventions: List[Dict],
                 divergence_threshold: float = 0.1,
                 min_training_steps: int = 50000,
                 max_training_steps: int = 200000,
                 reference_states_size: int = 1000,
                 patience: int = 3):
        """
        Args:
            interventions: List of intervention configurations
            divergence_threshold: KL divergence threshold to advance curriculum
            min_training_steps: Minimum steps before checking divergence
            max_training_steps: Maximum steps per intervention
            reference_states_size: Number of reference states for comparison
            patience: Number of checks below threshold before advancing
        """
        self.interventions = interventions
        self.current_intervention_idx = 0
        self.divergence_threshold = divergence_threshold
        self.min_training_steps = min_training_steps
        self.max_training_steps = max_training_steps
        self.reference_states_size = reference_states_size
        self.patience = patience
        
        # State tracking
        self.reference_states = None
        self.policy_before = None
        self.steps_in_current_intervention = 0
        self.divergence_history = []
        self.intervention_history = []
        self.patience_counter = 0
        
    def should_advance_curriculum(self, model: PPO, replay_buffer_states: np.ndarray) -> bool:
        """
        Determine if curriculum should advance based on policy divergence.
        """
        # Don't check until minimum training steps
        if self.steps_in_current_intervention < self.min_training_steps:
            return False
            
        # Force advance if maximum steps reached
        if self.steps_in_current_intervention >= self.max_training_steps:
            print(f"Max steps reached for intervention {self.current_intervention_idx}")
            return True
            
        # Sample reference states if not set
        if self.reference_states is None:
            self.reference_states = self._sample_reference_states(replay_buffer_states)
            # Save policy snapshot before intervention
            self.policy_before = self._copy_policy(model)
            return False
            
        # Calculate policy divergence
        divergence = PolicyDivergenceCalculator.compute_policy_divergence(
            self.policy_before, model, self.reference_states
        )
        
        self.divergence_history.append({
            'intervention': self.current_intervention_idx,
            'step': self.steps_in_current_intervention,
            'divergence': divergence
        })
        
        print(f"Intervention {self.current_intervention_idx}, Step {self.steps_in_current_intervention}: "
              f"KL Divergence = {divergence:.4f}, Threshold = {self.divergence_threshold}")
        
        # Check if divergence is below threshold
        if divergence < self.divergence_threshold:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                print(f"Advancing curriculum: divergence {divergence:.4f} < {self.divergence_threshold} "
                      f"for {self.patience} consecutive checks")
                return True
        else:
            self.patience_counter = 0
            
        return False
    
    def advance_intervention(self, model: PPO, replay_buffer_states: np.ndarray):
        """Advance to the next intervention."""
        # Record intervention completion
        self.intervention_history.append({
            'intervention_idx': self.current_intervention_idx,
            'intervention_type': self.interventions[self.current_intervention_idx]['type'],
            'steps_trained': self.steps_in_current_intervention,
            'final_divergence': self.divergence_history[-1]['divergence'] if self.divergence_history else None
        })
        
        # Move to next intervention
        self.current_intervention_idx += 1
        self.steps_in_current_intervention = 0
        self.patience_counter = 0
        
        # Sample new reference states and save policy snapshot
        if self.current_intervention_idx < len(self.interventions):
            self.reference_states = self._sample_reference_states(replay_buffer_states)
            self.policy_before = self._copy_policy(model)
        
        print(f"Advanced to intervention {self.current_intervention_idx}")
    
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
        # Save current model to temporary location
        temp_path = "/tmp/temp_policy.zip"
        model.save(temp_path)
        # Load copy
        policy_copy = PPO.load(temp_path)
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return policy_copy
    
    def get_current_intervention(self) -> Optional[Dict]:
        """Get current intervention configuration."""
        if self.current_intervention_idx >= len(self.interventions):
            return None
        return self.interventions[self.current_intervention_idx]
    
    def is_curriculum_complete(self) -> bool:
        """Check if all interventions have been completed."""
        return self.current_intervention_idx >= len(self.interventions)
    
    def step(self):
        """Increment step counter."""
        self.steps_in_current_intervention += 1
    
    def save_progress(self, filepath: str):
        """Save curriculum progress."""
        progress = {
            'current_intervention_idx': self.current_intervention_idx,
            'steps_in_current_intervention': self.steps_in_current_intervention,
            'divergence_history': self.divergence_history,
            'intervention_history': self.intervention_history,
            'reference_states': self.reference_states
        }
        with open(filepath, 'wb') as f:
            pickle.dump(progress, f)
    
    def load_progress(self, filepath: str):
        """Load curriculum progress."""
        with open(filepath, 'rb') as f:
            progress = pickle.load(f)
        
        self.current_intervention_idx = progress['current_intervention_idx']
        self.steps_in_current_intervention = progress['steps_in_current_intervention']
        self.divergence_history = progress['divergence_history']
        self.intervention_history = progress['intervention_history']
        self.reference_states = progress['reference_states']

class AutoCaLCEnvironment:
    """
    Environment wrapper that applies interventions directly using CausalWorld's intervention system.
    """
    
    def __init__(self, base_env: CausalWorld, curriculum_manager: AutoCaLCCurriculumManager):
        self.base_env = base_env
        self.curriculum_manager = curriculum_manager
        
        # Create intervention actors
        self.intervention_actors = []
        for intervention in curriculum_manager.interventions:
            if intervention['type'] == 'goal':
                actor = GoalInterventionActorPolicy()
            elif intervention['type'] == 'mass':
                actor = PhysicalPropertiesInterventionActorPolicy(group='mass')
            elif intervention['type'] == 'friction':
                actor = PhysicalPropertiesInterventionActorPolicy(group='friction')
            elif intervention['type'] == 'visual':
                actor = VisualInterventionActorPolicy()
            else:
                raise ValueError(f"Unknown intervention type: {intervention['type']}")
            
            # Initialize the actor with the environment
            actor.initialize(base_env)
            self.intervention_actors.append(actor)
        
        self.episode_count = 0
        
    def reset(self):
        """Reset environment and apply current intervention."""
        obs = self.base_env.reset()
        self.episode_count += 1
        
        # Apply current intervention if active
        current_intervention = self.curriculum_manager.get_current_intervention()
        if current_intervention is not None:
            current_idx = self.curriculum_manager.current_intervention_idx
            if current_idx < len(self.intervention_actors):
                # Apply intervention using CausalWorld's intervention system
                intervention_dict = self.intervention_actors[current_idx]._act(
                    self.base_env.get_variable_space_used()
                )
                if intervention_dict:
                    self.base_env.do_intervention(intervention_dict)
                    print(f"Applied intervention {current_idx}: {current_intervention['type']}")
        
        return obs
    
    def step(self, action):
        """Step the environment."""
        return self.base_env.step(action)
    
    def advance_curriculum_if_needed(self, model: PPO, replay_buffer_states: np.ndarray):
        """Check and advance curriculum if conditions are met."""
        if self.curriculum_manager.should_advance_curriculum(model, replay_buffer_states):
            self.curriculum_manager.advance_intervention(model, replay_buffer_states)
            return True
        return False
    
    def __getattr__(self, name):
        """Delegate attribute access to base environment."""
        return getattr(self.base_env, name)

class AutoCaLCCallback(BaseCallback):
    """Callback to manage AutoCaLC curriculum during training."""
    
    def __init__(self, curriculum_manager: AutoCaLCCurriculumManager, 
                 check_freq: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.curriculum_manager = curriculum_manager
        self.check_freq = check_freq
        self.replay_buffer_states = deque(maxlen=10000)
        
    def _on_step(self) -> bool:
        # Collect states for replay buffer
        if hasattr(self.locals, 'obs_tensor'):
            obs = self.locals['obs_tensor'].cpu().numpy()
            self.replay_buffer_states.extend(obs)
        
        # Increment curriculum step counter
        self.curriculum_manager.step()
        
        # Check curriculum advancement
        if self.n_calls % self.check_freq == 0 and len(self.replay_buffer_states) > 0:
            replay_states = np.array(list(self.replay_buffer_states))
            
            # Check if we should advance curriculum
            if hasattr(self.training_env, 'envs'):
                # For vectorized environments, use first environment
                env = self.training_env.envs[0]
                if hasattr(env, 'advance_curriculum_if_needed'):
                    advanced = env.advance_curriculum_if_needed(self.model, replay_states)
                    if advanced:
                        # Log curriculum advancement
                        wandb.log({
                            'curriculum/intervention_idx': self.curriculum_manager.current_intervention_idx,
                            'curriculum/advancement_step': self.num_timesteps
                        }, step=self.num_timesteps)
        
        # Log divergence metrics
        if self.curriculum_manager.divergence_history:
            latest_divergence = self.curriculum_manager.divergence_history[-1]
            wandb.log({
                'curriculum/kl_divergence': latest_divergence['divergence'],
                'curriculum/current_intervention': self.curriculum_manager.current_intervention_idx,
                'curriculum/intervention_steps': self.curriculum_manager.steps_in_current_intervention
            }, step=self.num_timesteps)
        
        return True

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
                'curriculum_type': 'AutoCaLC',
                'max_episode_length': max_episode_length,
                'skip_frame': skip_frame,
                'seed': seed,
                'total_timesteps': total_timesteps,
                **ppo_config,
                **curriculum_config
            },
            tags=[task_name, 'PPO', 'AutoCaLC', 'curriculum'],
            sync_tensorboard=True
        )
    
    os.makedirs(log_dir, exist_ok=True)
    
    # Create curriculum manager
    curriculum_manager = AutoCaLCCurriculumManager(
        interventions=curriculum_config['interventions'],
        divergence_threshold=curriculum_config.get('divergence_threshold', 0.1),
        min_training_steps=curriculum_config.get('min_training_steps', 50000),
        max_training_steps=curriculum_config.get('max_training_steps', 200000),
        reference_states_size=curriculum_config.get('reference_states_size', 1000),
        patience=curriculum_config.get('patience', 3)
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
        name_prefix='autocalc_model'
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
        tb_log_name="autocalc_curriculum"
    )
    
    # Save final model and curriculum progress
    model.save(os.path.join(log_dir, 'final_autocalc_model'))
    curriculum_manager.save_progress(os.path.join(log_dir, 'curriculum_progress.pkl'))
    
    # Generate curriculum analysis plots
    plot_curriculum_analysis(curriculum_manager, log_dir)
    
    if wandb_config:
        wandb.finish()
    
    print(f"AutoCaLC training completed. Model saved to {log_dir}")

def plot_curriculum_analysis(curriculum_manager: AutoCaLCCurriculumManager, save_dir: str):
    """Generate analysis plots for curriculum learning."""
    
    # Plot 1: KL Divergence over time
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    if curriculum_manager.divergence_history:
        steps = [d['step'] for d in curriculum_manager.divergence_history]
        divergences = [d['divergence'] for d in curriculum_manager.divergence_history]
        interventions = [d['intervention'] for d in curriculum_manager.divergence_history]
        
        plt.plot(steps, divergences, 'b-', alpha=0.7)
        plt.axhline(y=curriculum_manager.divergence_threshold, color='r', linestyle='--', 
                   label=f'Threshold ({curriculum_manager.divergence_threshold})')
        
        # Mark intervention changes
        for i, intervention in enumerate(curriculum_manager.intervention_history):
            plt.axvline(x=intervention['steps_trained'], color='g', linestyle=':', alpha=0.7)
        
        plt.xlabel('Training Steps')
        plt.ylabel('KL Divergence')
        plt.title('Policy Divergence During Curriculum')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot 2: Intervention timeline
    plt.subplot(2, 2, 2)
    if curriculum_manager.intervention_history:
        intervention_types = [h['intervention_type'] for h in curriculum_manager.intervention_history]
        steps_trained = [h['steps_trained'] for h in curriculum_manager.intervention_history]
        
        plt.barh(range(len(intervention_types)), steps_trained)
        plt.yticks(range(len(intervention_types)), intervention_types)
        plt.xlabel('Training Steps')
        plt.title('Time Spent per Intervention')
        plt.grid(True, alpha=0.3)
    
    # Plot 3: Final divergences per intervention
    plt.subplot(2, 2, 3)
    if curriculum_manager.intervention_history:
        final_divergences = [h.get('final_divergence', 0) for h in curriculum_manager.intervention_history]
        intervention_types = [h['intervention_type'] for h in curriculum_manager.intervention_history]
        
        plt.bar(range(len(intervention_types)), final_divergences)
        plt.xticks(range(len(intervention_types)), intervention_types, rotation=45)
        plt.ylabel('Final KL Divergence')
        plt.title('Final Divergence per Intervention')
        plt.grid(True, alpha=0.3)
    
    # Plot 4: Curriculum progression
    plt.subplot(2, 2, 4)
    if curriculum_manager.divergence_history:
        # Color points by intervention
        colors = plt.cm.Set1(np.linspace(0, 1, len(curriculum_manager.interventions)))
        for i, intervention in enumerate(curriculum_manager.interventions):
            intervention_data = [d for d in curriculum_manager.divergence_history if d['intervention'] == i]
            if intervention_data:
                steps = [d['step'] for d in intervention_data]
                divergences = [d['divergence'] for d in intervention_data]
                plt.scatter(steps, divergences, c=[colors[i]], label=intervention['type'], alpha=0.6)
        
        plt.xlabel('Steps in Intervention')
        plt.ylabel('KL Divergence')
        plt.title('Divergence by Intervention Type')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'autocalc_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='AutoCaLC Curriculum Learning for CausalWorld')
    parser.add_argument('--train', action='store_true', help='Train with AutoCaLC curriculum')
    parser.add_argument('--task', type=str, default='pushing', 
                       choices=list(DENSE_REWARD_WEIGHTS.keys()), help='Task name')
    parser.add_argument('--pretrained_path', type=str, 
                       help='Path to pretrained PPO model')
    parser.add_argument('--log_dir', type=str, default='autocalc_curriculum', help='Log directory')
    parser.add_argument('--timesteps', type=int, default=2_000_000, help='Total timesteps')
    parser.add_argument('--num_envs', type=int, default=16, help='Number of parallel environments')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--skip_frame', type=int, default=3, help='Frame skip')
    parser.add_argument('--max_episode_length', type=int, default=250, help='Max episode length')
    parser.add_argument('--eval_freq', type=int, default=100_000, help='Evaluation frequency')
    parser.add_argument('--use_wandb', action='store_true', help='Enable wandb logging')
    
    # AutoCaLC specific parameters
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
    
    args = parser.parse_args()
    
    # Auto-determine pretrained path if not provided
    if args.pretrained_path is None:
        args.pretrained_path = f'ppo_{args.task}_sb3/final_model.zip'
        print(f"Auto-determined pretrained path: {args.pretrained_path}")
    
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
    
    # Curriculum configuration
    curriculum_config = {
        'interventions': [
            {'type': 'goal'},      # Start with goal position variations
            {'type': 'mass'},      # Then object mass variations
            {'type': 'friction'},  # Then friction variations
            {'type': 'visual'},    # Finally visual variations
        ],
        'divergence_threshold': args.divergence_threshold,
        'min_training_steps': args.min_training_steps,
        'max_training_steps': args.max_training_steps,
        'reference_states_size': args.reference_states_size,
        'patience': args.patience
    }
    
    wandb_config = None
    if args.use_wandb:
        wandb_config = {
            'project': f'autocalc-curriculum-{args.task}',
            'run_name': f'autocalc_{args.task}_thresh{args.divergence_threshold}_seed{args.seed}'
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
