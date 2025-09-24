"""
AutoCaLC (meta-teacher/student) runner.

Summary
- Iterate over M meta-episodes:
  1) compute meta-state from the student,
  2) teacher picks an intervention,
  3) student trains K steps on that intervention,
  4) evaluate on a fixed validation env,
  5) meta-reward = delta(validation), update teacher.

TRAINING COMMANDS:

Basic training (recommended start):
python meta_teacher_student.py --task pushing --meta_episodes 50 --student_train_steps 5000 --eval --use_wandb --device_id 7

Full training with custom parameters:
python meta_teacher_student.py --task pushing --meta_episodes 100 --student_train_steps 10000 \
    --teacher_lr 1e-4 --teacher_epsilon 0.3 --teacher_min_epsilon 0.05 --teacher_epsilon_decay 0.90 \
    --sequence_length 10 --count_beta 0.01 --teacher_state_cm_weight 0.5 \
    --validation_episodes 10 --intervention_test_episodes 5 \
    --use_wandb --device_id 0 --log_dir logs/autocalc_pushing_exp1 --eval

Quick test run (fast debugging):
python meta_teacher_student.py --task pushing --meta_episodes 5 --student_train_steps 1000 --validation_episodes 3

Multi-task training examples:
python meta_teacher_student.py --task reaching --meta_episodes 30 --student_train_steps 3000 --eval --device_id 1
python meta_teacher_student.py --task picking --meta_episodes 75 --student_train_steps 7500 --eval --device_id 2
python meta_teacher_student.py --task stacking2 --meta_episodes 100 --student_train_steps 10000 --eval --device_id 3

EVALUATION COMMANDS:

Evaluate trained model:
python meta_teacher_student.py --eval_only --log_dir logs/autocalc_logs --task pushing --eval_episodes 20

Evaluate with custom model path:
python meta_teacher_student.py --eval_only --eval_model_path logs/my_experiment/best_student_model.zip \
    --task pushing --eval_episodes 20 --max_episode_length 300

PARAMETER TUNING GUIDE:
- Teacher exploration: --teacher_epsilon 0.1-0.5, --teacher_min_epsilon 0.01-0.1, --teacher_epsilon_decay 0.8-0.95
- Count-based exploration: --count_beta 0.001-0.1, --count_encoding_dim 16-64
- Teacher state: --teacher_state_cm_weight 0.0-1.0 (0.0=reward only, 1.0=CM only)
- Memory: --sequence_length 3-15 (LSTM history length)
- Testing: --intervention_test_episodes 3-10, --validation_episodes 5-15

SUPPORTED TASKS: pushing, reaching, picking, pick_and_place, stacking2

For all options: python meta_teacher_student.py --help
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import json
import argparse
import logging
from collections import deque
from copy import deepcopy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
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
from causal_world.task_generators import generate_task

# importing the resuable components from baselines and validation_actor
# because why not
from baselines import (
    INTERVENTIONS,
    TASK_BENCHMARKS,
    SUPPORTED_TASKS,
    DENSE_REWARD_WEIGHTS,
    create_environment,
    evaluate_cm_score,
    test_intervention_performance,
    test_all_interventions,
    run_post_episode_validation,
    RewardMonitorCallback,
    IntervenedCausalWorld,
    CSVLogger,
    ValidationCallback,
    CountBasedRewardCallback
)
from validation_actor import ValidationInterventionActorPolicy
import wandb
from wandb.integration.sb3 import WandbCallback


class RecurrentDQN(nn.Module):
    """DQN with LSTM for tracking past trajectories and adapting to student performance"""
    def __init__(self, state_dim, action_dim, hidden_dim=128, lstm_layers=2):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers

        # Input processing layer
        self.input_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # LSTM for sequential learning
        # KAUSAR: understand how LSTMs work
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, lstm_layers, batch_first=True, dropout=0.1)

        # Q-value head (removed dueling architecture)
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, states, hidden=None):
        """
        states: (batch_size, seq_length, state_dim) or (batch_size, state_dim)
        hidden: LSTM hidden state tuple (h_0, c_0)
        """
        if len(states.shape) == 2:  # Single timestep
            states = states.unsqueeze(1)
        
        batch_size, seq_len, _ = states.shape
        
        # Process input through initial network
        processed = self.input_net(states.view(-1, self.state_dim))
        processed = processed.view(batch_size, seq_len, self.hidden_dim)
        
        # LSTM forward pass
        lstm_out, new_hidden = self.lstm(processed, hidden)
        
        # Use last timestep output for Q-value computation
        last_output = lstm_out[:, -1, :]
        
        q_values = self.q_head(last_output)
        
        return q_values, new_hidden

class RecurrentTeacherAgent:
    """Teacher DQN agent with LSTM memory and logging"""
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, device='cpu',
                 buffer_size=10000, batch_size=32, sequence_length=5, target_update_freq=100,
                 min_epsilon=0.05, epsilon_decay=0.90):
        self.device = device
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.update_count = 0
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

        # Networks with simple DQN architecture
        self.q_net = RecurrentDQN(state_dim, action_dim).to(device)
        self.target_net = deepcopy(self.q_net)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        # Memory and state tracking
        self.replay_buffer = ReplayBuffer(buffer_size, state_dim, device, sequence_length)
        self.state_history = deque(maxlen=sequence_length)
        self.hidden_state = None
        
        # Training statistics
        self.training_stats = {
            'losses': [],
            'q_values': [],
            'actions_taken': [],
            'rewards_received': [],
            'epsilon_values': []
        }
    
    def select_action(self, state, epsilon=0.1, meta_step=0):
        """Select action using epsilon-greedy with LSTM context and adaptive exploration"""
        # Add current state to history
        self.state_history.append(state)
        
        # Adaptive epsilon decay based on meta_step
        adaptive_epsilon = max(self.min_epsilon, epsilon * (self.epsilon_decay ** meta_step))
        self.training_stats['epsilon_values'].append(adaptive_epsilon)
        
        if random.random() < adaptive_epsilon:
            action = random.randint(0, self.q_net.action_dim - 1)
            self.training_stats['actions_taken'].append(f"random_{action}")
            return action

        # Prepare sequence for LSTM
        if len(self.state_history) < self.sequence_length:
            # Pad with zeros if not enough history
            padded_states = [np.zeros_like(state) for _ in range(self.sequence_length - len(self.state_history))]
            padded_states.extend(list(self.state_history))
            sequence = np.stack(padded_states)
        else:
            sequence = np.stack(list(self.state_history))
        
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values, self.hidden_state = self.q_net(sequence_tensor, self.hidden_state)
            action = int(q_values.argmax().item())
            
            # Log Q-values for analysis
            self.training_stats['q_values'].append(q_values.cpu().numpy().flatten())
            self.training_stats['actions_taken'].append(f"greedy_{action}")
        
        return action

    def train_step(self, meta_step=0):
        """Enhanced training step with Double DQN and comprehensive logging"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        state_seqs, actions, rewards, next_state_seqs = self.replay_buffer.sample(self.batch_size)
        
        # Current Q-values
        current_q, _ = self.q_net(state_seqs)
        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        # DQN target calculation
        with torch.no_grad():
            next_q_values, _ = self.target_net(next_state_seqs)
            max_next_q_values = next_q_values.max(dim=1)[0]
            target_q = rewards + self.gamma * max_next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        # Update target network periodically
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Log training statistics
        self.training_stats['losses'].append(loss.item())
        self.training_stats['rewards_received'].extend(rewards.cpu().numpy().tolist())

        return loss.item()

    def update_state_history(self, state):
        """update state history for recurrent processing"""
        self.state_history.append(state)
        if len(self.state_history) > self.sequence_length:
            self.state_history.popleft()

    def get_training_stats(self):
        """Return comprehensive training statistics"""
        if not self.training_stats['losses']:
            return {}
        
        return {
            'avg_loss': np.mean(self.training_stats['losses'][-100:]),  # Last 100 steps
            'avg_reward': np.mean(self.training_stats['rewards_received'][-100:]),
            'avg_q_value': np.mean([np.mean(q) for q in self.training_stats['q_values'][-100:]]),
            'current_epsilon': self.training_stats['epsilon_values'][-1] if self.training_stats['epsilon_values'] else 0,
            'total_updates': self.update_count
        }

    def save(self, path):
        """Save teacher model and training statistics"""
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'update_count': self.update_count,
            'state_history': list(self.state_history)
        }, path)
        logging.info(f"Teacher model and stats saved to {path}")
    
    def load(self, path):
        """Load teacher model and training statistics"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No teacher model found at {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'training_stats' in checkpoint:
            self.training_stats = checkpoint['training_stats']
        if 'update_count' in checkpoint:
            self.update_count = checkpoint['update_count']
        if 'state_history' in checkpoint:
            self.state_history = deque(checkpoint['state_history'], maxlen=self.sequence_length)
        
        logging.info(f"Teacher model loaded from {path}")


class ReplayBuffer:
    """replay buffer for storing sequences and experiences"""
    def __init__(self, capacity, state_dim, device, sequence_length=5):
        self.capacity = capacity
        self.device = device
        self.sequence_length = sequence_length
        self.buffer = deque(maxlen=capacity)
        self.state_sequences = deque(maxlen=capacity)
    
    def push(self, state_sequence, action, reward, next_state_sequence):
        """store a complete experience with state sequences"""
        self.buffer.append((state_sequence, action, reward, next_state_sequence))
        self.state_sequences.append(state_sequence)

    def sample(self, batch_size):
        """sample batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        state_seqs, actions, rewards, next_state_seqs = zip(*batch)

        return (
            torch.tensor(np.stack(state_seqs), dtype=torch.float32).to(self.device),
            torch.tensor(actions, dtype=torch.long).to(self.device),
            torch.tensor(rewards, dtype=torch.float32).to(self.device),
            torch.tensor(np.stack(next_state_seqs), dtype=torch.float32).to(self.device)
        )
    
    def __len__(self):
        return len(self.buffer)


# get the teacher state
def get_teacher_state(student_model, task_name, interventions, device='cpu', seed=0, 
                     test_episodes=5, cm_weight=0.5):
    """
    Compute the meta-state for the teacher by evaluating reward and CM score for each intervention
    Returns a combined score array: [cm_weight*cm + (1-cm_weight)*reward for each intervention]
    """
    logging.info("Computing teacher state (rewards and CM scores for all interventions)")
    cm_scores = []
    rewards = []

    for i, intervention in enumerate(interventions):
        # 1. Test the reward performance
        reward_metrics = test_intervention_performance(
            student_model,
            intervention,
            task_name,
            num_episodes=test_episodes,
            seed=seed + i
        )
        rewards.append(reward_metrics['avg_reward'])

        # 2. Compute CM score
        env = create_environment(task_name, intervention, seed=seed + i + 100)
        cm_score = evaluate_cm_score(
            env, 
            student_model, 
            max_episodes=test_episodes, 
            device=device,
            intervention_type=intervention['type'], 
            seed=seed + i + 100
        )
        
        cm_scores.append(cm_score)
        env.close()
    
    # normalize scores
    reward_array = np.array(rewards, dtype=np.float32)
    cm_array = np.array(cm_scores, dtype=np.float32)

    # simple normalization
    normalized_rewards = (reward_array - np.mean(reward_array)) / (np.std(reward_array) + 1e-8)
    normalized_cm = (cm_array - np.mean(cm_array)) / (np.std(cm_array) + 1e-8)

    # combine rewards and CM scores with configurable weight
    state = cm_weight * normalized_cm + (1 - cm_weight) * normalized_rewards

    return state

# evaluate student generalization
def run_validation_protocol(student_model, task_name, stage_num, args, csv_logger, cumulative_timesteps):
    """Evaluate generalization performance on the 10, fixed validation environments"""
    logging.info(f"[VALIDATION] Running validation after meta-episode {stage_num}")

    # create validation callback with the same settings as baselines
    validation_callback = ValidationCallback(
        validation_frequency = float('inf'),    # only run when explicitly called
        task_name=task_name,
        csv_logger=csv_logger,
        stage=stage_num,
        cumulative_timesteps=cumulative_timesteps,
        validation_episodes=args.validation_episodes,
        seed=args.seed + stage_num * 10000,
        baseline_type="autocalc"
    )
    
    # run validation and return metrics
    validation_metrics = validation_callback._execute_validation(student_model)

    return validation_metrics

class AutoCaLC:
    """Enhanced AutoCaLC with adaptive RNN-based teacher and comprehensive logging"""
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
        
        # Initialize logging
        self.setup_logging()
        
        # Initialize models
        self.setup_models()
        
        # Best model tracking
        self.best_validation_reward = float('-inf')
        self.best_model_meta_step = 0
        self.best_model_path = os.path.join(args.log_dir, "best_student_model.zip")
        
        # Training statistics
        self.training_stats = {
            'meta_rewards': [],
            'validation_rewards': [],
            'validation_success_rates': [],
            'teacher_losses': [],
            'selected_interventions': [],
            'cumulative_timesteps': []
        }

    def setup_logging(self):
        """Setup comprehensive logging system"""
        os.makedirs(self.args.log_dir, exist_ok=True)
        
        # Clear existing handlers to prevent conflicts with imported modules
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.args.log_dir, 'autocalc_training.log')),
                logging.StreamHandler()
            ]
        )
        
        # Initialize CSV logger
        self.csv_logger = CSVLogger(self.args.log_dir)
        
        # Initialize wandb if enabled
        if self.args.use_wandb:
            wandb.init(
                project=f'autocalc-{self.args.task}',
                name=f'autocalc_{self.args.task}_seed{self.args.seed}',
                config=vars(self.args),
                tags=[self.args.task, 'autocalc', 'recurrent-teacher']
            )

    def setup_models(self):
        """Initialize student and teacher models"""
        # Initialize student (PPO)
        env = create_environment(self.args.task, None, seed=self.args.seed)
        self.student = PPO("MlpPolicy", env, verbose=0, device=self.device)
        env.close()

        # Initialize teacher (recurrent DQN)
        state_dim = len(INTERVENTIONS)  # rewards + CM scores
        action_dim = len(INTERVENTIONS)
        self.teacher = RecurrentTeacherAgent(
            state_dim, 
            action_dim, 
            device=self.device,
            lr=self.args.teacher_lr,
            gamma=self.args.teacher_gamma,
            buffer_size=self.args.teacher_buffer_size,
            batch_size=self.args.teacher_batch_size,
            sequence_length=self.args.sequence_length,
            min_epsilon=self.args.teacher_min_epsilon,
            epsilon_decay=self.args.teacher_epsilon_decay
        )
        
        logging.info(f"Models initialized on device: {self.device}")

    def update_best_model(self, validation_reward, meta_step):
        """Update best model if current performance is better"""
        if validation_reward > self.best_validation_reward:
            self.best_validation_reward = validation_reward
            self.best_model_meta_step = meta_step
            
            # Save as best model
            self.student.save(self.best_model_path)
            logging.info(f"New best model at meta-step {meta_step} with validation reward: {validation_reward:.4f}")
            
            # Log to wandb if enabled
            if self.args.use_wandb:
                wandb.log({
                    'best_validation_reward': validation_reward,
                    'best_model_meta_step': meta_step
                })
            
            return True
        return False

    def save_periodic_models(self, meta_step):
        """Save models periodically every 10 meta-episodes"""
        if meta_step % 10 == 0:
            # Save student model
            student_path = os.path.join(self.args.log_dir, f"student_model_step_{meta_step}.zip")
            self.student.save(student_path)
            
            # Save teacher model
            teacher_path = os.path.join(self.args.log_dir, f"teacher_model_step_{meta_step}.pt")
            self.teacher.save(teacher_path)
            
            logging.info(f"\n[MODEL CHECKPOINT] Saved models at meta-step {meta_step}")
            logging.info(f"  Student: {student_path}")
            logging.info(f"  Teacher: {teacher_path}")
            logging.info(f"  Current validation reward: {self.training_stats['validation_rewards'][-1]:.4f}")
            logging.info(f"  Best validation reward so far: {self.best_validation_reward:.4f}")

    def log_training_progress(self, meta_step, meta_reward, validation_metrics, teacher_stats, 
                             selected_intervention, cumulative_timesteps, count_callback=None):
        """Comprehensive logging of training progress"""
        # Update training statistics
        self.training_stats['meta_rewards'].append(meta_reward)
        self.training_stats['validation_rewards'].append(validation_metrics['validation_avg_reward'])
        self.training_stats['validation_success_rates'].append(validation_metrics['validation_success_rate'])
        self.training_stats['selected_interventions'].append(selected_intervention['type'])
        
        if teacher_stats.get('avg_loss'):
            self.training_stats['teacher_losses'].append(teacher_stats['avg_loss'])

        # Log to console
        logging.info(f"\n[TRAINING SUMMARY] Meta-step {meta_step}/{self.args.meta_episodes}")
        logging.info(f"  Intervention: {selected_intervention['type']} | Meta-reward: {meta_reward:.4f}")
        logging.info(f"  Validation: {validation_metrics['validation_avg_reward']:.4f} (±{validation_metrics['validation_reward_std']:.3f})")
        logging.info(f"  Success rate: {validation_metrics['validation_success_rate']:.3f}")
        
        # Handle teacher stats which might be empty or None
        current_epsilon = teacher_stats.get('current_epsilon', 0)
        avg_q_value = teacher_stats.get('avg_q_value', 0)
        if current_epsilon is not None and avg_q_value is not None:
            logging.info(f"  Teacher ε: {current_epsilon:.4f} | Q-avg: {avg_q_value:.3f}")
        else:
            logging.info(f"  Teacher ε: {current_epsilon or 'N/A'} | Q-avg: {avg_q_value or 'N/A'}")
        
        # Show intervention preference trend (last 5 interventions)
        if len(self.training_stats['selected_interventions']) >= 5:
            recent_interventions = self.training_stats['selected_interventions'][-5:]
        else:
            recent_interventions = self.training_stats['selected_interventions']
        logging.info(f"  Recent choices: {' -> '.join(recent_interventions)}")
        
        # Best model status
        if validation_metrics['validation_avg_reward'] > self.best_validation_reward:
            logging.info(f"  🌟 NEW BEST MODEL! Previous best: {self.best_validation_reward:.4f}")

        # Log to wandb
        if self.args.use_wandb:
            # Safely get teacher stats
            teacher_epsilon = 0
            if (hasattr(self.teacher, 'training_stats') and 
                'epsilon_values' in self.teacher.training_stats and 
                self.teacher.training_stats['epsilon_values']):
                teacher_epsilon = self.teacher.training_stats['epsilon_values'][-1]
            
            wandb_log = {
                'meta_step': meta_step,
                'meta_reward': meta_reward,
                'validation/reward': validation_metrics['validation_avg_reward'],
                'validation/success_rate': validation_metrics['validation_success_rate'],
                'validation/reward_std': validation_metrics['validation_reward_std'],
                'selected_intervention': selected_intervention['type'],
                'teacher/epsilon': teacher_epsilon,
                'teacher/avg_q_value': teacher_stats.get('avg_q_value', 0) or 0,
                'teacher/loss': teacher_stats.get('avg_loss', 0) or 0,
                'cumulative_timesteps': cumulative_timesteps,
                'unique_states': len(count_callback.visit_counts) if count_callback else 0,
            }
            
            # Add teacher stats safely
            for k, v in teacher_stats.items():
                if v is not None and isinstance(v, (int, float)):
                    wandb_log[f"teacher_{k}"] = v
                    
            wandb.log(wandb_log)

    def train(self):
        """Main training loop with enhanced logging and adaptive teacher"""
        logging.info("Starting AutoCaLC training with recurrent teacher...")
        
        # add test episodes attribute for initial validation
        self.args.test_episodes = self.args.validation_episodes
        
        # Initial validation
        initial_validation = run_validation_protocol(
            self.student, self.args.task, 0, self.args, self.csv_logger, 0
        )
        last_validation_reward = initial_validation['validation_avg_reward']
        
        # Update best model with initial performance
        self.update_best_model(last_validation_reward, 0)
        
        cumulative_timesteps = 0

        for meta_step in range(1, self.args.meta_episodes + 1):
            logging.info(f"\n{'='*60}")
            logging.info(f"Meta-Episode {meta_step}/{self.args.meta_episodes}")
            logging.info(f"{'='*60}")

            # 1. Get current meta-state
            current_meta_state = get_teacher_state(
                self.student, self.args.task, INTERVENTIONS,
                device=self.device, seed=self.args.seed + meta_step,
                test_episodes=self.args.intervention_test_episodes,
                cm_weight=self.args.teacher_state_cm_weight
            )

            self.teacher.update_state_history(current_meta_state)

            # 2. Teacher selects intervention (with adaptive exploration)
            selected_intervention_idx = self.teacher.select_action(
                current_meta_state, 
                epsilon=self.args.teacher_epsilon,
                meta_step=meta_step
            )
            selected_intervention = INTERVENTIONS[selected_intervention_idx]

            # Log teacher's decision criteria
            logging.info(f"\n[TEACHER DECISION] Meta-step {meta_step}")
            logging.info(f"  State representation: {current_meta_state}")
            
            # Handle Q-values which might be empty
            if self.teacher.training_stats['q_values']:
                q_vals = self.teacher.training_stats['q_values'][-1]
                logging.info(f"  Q-values for interventions: {q_vals}")
            else:
                logging.info(f"  Q-values for interventions: N/A (no Q-values computed yet)")
                
            logging.info(f"  Selected: {selected_intervention['type']} (idx: {selected_intervention_idx})")
            
            # Handle epsilon values which might be empty
            if self.teacher.training_stats['epsilon_values']:
                current_epsilon = self.teacher.training_stats['epsilon_values'][-1]
                logging.info(f"  Exploration rate: {current_epsilon:.4f}")
            else:
                logging.info(f"  Exploration rate: N/A")
                
            # Handle action types which might be empty
            if self.teacher.training_stats['actions_taken']:
                decision_type = self.teacher.training_stats['actions_taken'][-1]
                logging.info(f"  Decision type: {decision_type}")
            else:
                logging.info(f"  Decision type: N/A")

            # 3. Train student on selected intervention
            train_env = create_environment(self.args.task, selected_intervention, seed=self.args.seed + meta_step)
            vec_env = DummyVecEnv([lambda: train_env])
            vec_env = VecMonitor(vec_env, filename=os.path.join(self.args.log_dir, f'monitor_stage{meta_step}.csv'))

            # Add count-based exploration callback
            count_callback = CountBasedRewardCallback(
                beta=self.args.count_beta,
                encoding_dim=self.args.count_encoding_dim,
                verbose=1
            )
            
            # Set up other callbacks
            reward_monitor = RewardMonitorCallback(
                selected_intervention['type'], self.csv_logger, meta_step, 
                cumulative_timesteps, baseline_type="autocalc"
            )
            
            callbacks = [count_callback, reward_monitor]
            if self.args.use_wandb:
                callbacks.append(WandbCallback(gradient_save_freq=1000, verbose=0))

            # Train student with count-based exploration
            self.student.set_env(vec_env)
            
            # Log pre-training performance
            pre_training_reward = current_validation['validation_avg_reward'] if 'current_validation' in locals() else last_validation_reward
            logging.info(f"\n[STUDENT TRAINING] Starting on {selected_intervention['type']} intervention...")
            logging.info(f"  Pre-training validation reward: {pre_training_reward:.4f}")
            
            self.student.learn(
                total_timesteps=self.args.student_train_steps,
                callback=CallbackList(callbacks),
                reset_num_timesteps=False
            )
            
            cumulative_timesteps += self.args.student_train_steps
            train_env.close()

            # Log post-training metrics
            logging.info(f"[STUDENT TRAINING] Completed {self.args.student_train_steps} steps")
            logging.info(f"  Unique states explored: {len(count_callback.visit_counts)}")
            logging.info(f"  Total cumulative timesteps: {cumulative_timesteps}")

            # x. run_post_episode_validation handles best model tracking
            # adding temporary test episodes attribute to match baselines.py expectation
            self.args.test_episodes = self.args.validation_episodes
            current_validation = run_post_episode_validation(
                student_model=self.student,
                task_name=self.args.task,
                stage_num=meta_step,
                args=self.args,
                csv_logger=self.csv_logger,
                cumulative_timesteps=cumulative_timesteps,
                baseline_type="autocalc",
                best_validation_info={
                    'best_reward': self.best_validation_reward,
                    'best_stage': self.best_model_meta_step,
                    'best_model_path': self.best_model_path,
                    'best_metrics': None
                }
            )
            meta_reward = current_validation['validation_avg_reward'] - last_validation_reward

            # update best info from return value
            if current_validation['validation_avg_reward'] > self.best_validation_reward:
                self.best_validation_reward = current_validation['validation_avg_reward']
                self.best_model_meta_step = meta_step

            # 7. Get next meta-state and train teacher
            next_meta_state = get_teacher_state(
                self.student, self.args.task, INTERVENTIONS,
                device=self.device, seed=self.args.seed + meta_step + 1,
                test_episodes=self.args.intervention_test_episodes,
                cm_weight=self.args.teacher_state_cm_weight
            )

            # Create state sequences for recurrent learning
            current_seq = list(self.teacher.state_history) if len(self.teacher.state_history) > 0 else [current_meta_state] * self.teacher.sequence_length
            next_seq = current_seq[1:] + [next_meta_state]  # Shift and add new state
            
            # Store experience and train teacher
            self.teacher.replay_buffer.push(
                np.array(current_seq), selected_intervention_idx, meta_reward, np.array(next_seq)
            )
            teacher_loss = self.teacher.train_step(meta_step)

            # Log teacher learning progress
            logging.info(f"\n[TEACHER LEARNING] Meta-reward: {meta_reward:.4f}")
            logging.info(f"  Validation improvement: {last_validation_reward:.4f} -> {current_validation['validation_avg_reward']:.4f}")
            
            # Handle teacher_loss which can be None
            if teacher_loss is not None:
                logging.info(f"  Teacher loss: {teacher_loss:.6f}")
            else:
                logging.info(f"  Teacher loss: N/A (insufficient replay buffer samples)")
                
            logging.info(f"  Replay buffer size: {len(self.teacher.replay_buffer)}")
            logging.info(f"  Target network updates: {self.teacher.update_count // self.teacher.target_update_freq}")

            # 8. Get teacher statistics, log progress, and leverage CSVLogger
            teacher_stats = self.teacher.get_training_stats()
            self.log_training_progress(meta_step, meta_reward, current_validation, teacher_stats, 
                                     selected_intervention, cumulative_timesteps, count_callback)
            
            # Log validation episode
            self.csv_logger.log_validation_episode(
                meta_step, cumulative_timesteps, current_validation['validation_avg_reward'],
                current_validation['validation_reward_std'], current_validation['validation_success_rate'],
                current_validation['validation_success_rate_std'], current_validation['validation_avg_length'],
                current_validation['validation_length_std'], baseline_type="autocalc"
            )

            # Test ALL interventions for comprehensive logging (like baselines)
            test_all_interventions(
                student_model=self.student,
                task_name=self.args.task,
                stage_num=meta_step,
                args=self.args,
                csv_logger=self.csv_logger,
                cumulative_timesteps=cumulative_timesteps,
                baseline_type="autocalc"
            )

            # Save models periodically
            self.save_periodic_models(meta_step)

            # Update for next iteration
            last_validation_reward = current_validation['validation_avg_reward']

        # Save final model and statistics
        final_model_path = os.path.join(self.args.log_dir, "final_student_model.zip")
        self.student.save(final_model_path)
        
        teacher_model_path = os.path.join(self.args.log_dir, "final_teacher_model.pt")
        self.teacher.save(teacher_model_path)

        # Training completion summary
        intervention_counts = {}
        for intervention in self.training_stats['selected_interventions']:
            intervention_counts[intervention] = intervention_counts.get(intervention, 0) + 1
        
        logging.info(f"\n{'='*80}")
        logging.info(f"TRAINING COMPLETED - AutoCaLC Summary")
        logging.info(f"{'='*80}")
        logging.info(f"Total meta-episodes: {self.args.meta_episodes}")
        logging.info(f"Total timesteps: {cumulative_timesteps:,}")
        logging.info(f"Final validation reward: {current_validation['validation_avg_reward']:.4f}")
        logging.info(f"Best validation reward: {self.best_validation_reward:.4f} (step {self.best_model_meta_step})")
        logging.info(f"Improvement: {self.best_validation_reward - initial_validation['validation_avg_reward']:.4f}")
        logging.info(f"\nIntervention Selection Frequency:")
        for intervention, count in sorted(intervention_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.training_stats['selected_interventions'])) * 100
            logging.info(f"  {intervention}: {count} times ({percentage:.1f}%)")
        
        logging.info(f"\nModel Files:")
        logging.info(f"  Final student: {final_model_path}")
        logging.info(f"  Best student: {self.best_model_path}")
        logging.info(f"  Final teacher: {teacher_model_path}")
        logging.info(f"{'='*80}")

        logging.info(f"Training completed!")
        logging.info(f"Best validation reward: {self.best_validation_reward:.4f} at meta-step {self.best_model_meta_step}")
        logging.info(f"Final model saved to: {final_model_path}")
        logging.info(f"Best model saved to: {self.best_model_path}")


        # aggregating results for final reporting
        final_results = {
            'best_validation_reward': self.best_validation_reward,
            'best_model_meta_step': self.best_model_meta_step,
            'final_validation_reward': current_validation['validation_avg_reward'],
            'total_timesteps': cumulative_timesteps,
            'meta_episodes': self.args.meta_episodes,
            'task': self.args.task
        }

        # save the results as a JSON
        results_path = os.path.join(self.args.log_dir, "autocalc_results.json")
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        logging.info(f"Final results saved to {results_path}")

        if self.args.use_wandb:
            wandb.run.summary.update(final_results)
            wandb.finish()

        return self.student, self.teacher


# Utility functions
def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed set to {seed}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="AutoCaLC: Meta-learning with Recurrent Teacher")
    
    # Basic training parameters
    parser.add_argument('--task', type=str, default='pushing', help='CausalWorld task to train on')
    parser.add_argument('--meta_episodes', type=int, default=50, help='Number of meta-episodes')
    parser.add_argument('--student_train_steps', type=int, default=5000, help='Timesteps for each student training block')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    
    # Teacher parameters
    parser.add_argument('--teacher_lr', type=float, default=1e-4, help='Teacher learning rate')
    parser.add_argument('--teacher_gamma', type=float, default=0.99, help='Teacher discount factor')
    parser.add_argument('--teacher_epsilon', type=float, default=0.3, help='Teacher exploration rate')
    parser.add_argument('--teacher_buffer_size', type=int, default=10000, help='Teacher replay buffer size')
    parser.add_argument('--teacher_batch_size', type=int, default=32, help='Teacher batch size')
    parser.add_argument('--sequence_length', type=int, default=5, help='LSTM sequence length')

    # Count-based exploration parameters for student
    parser.add_argument('--count_beta', type=float, default=0.01, help='Scaling factor for count-based intrinsic rewards')
    parser.add_argument('--count_encoding_dim', type=int, default=32, help='Encoding dimension for state hashing')
    
    # Teacher state and exploration parameters
    parser.add_argument('--intervention_test_episodes', type=int, default=5, help='Episodes for testing interventions')
    parser.add_argument('--teacher_state_cm_weight', type=float, default=0.5, help='Weight for CM score in teacher state')
    parser.add_argument('--teacher_min_epsilon', type=float, default=0.05, help='Minimum exploration rate for teacher')
    parser.add_argument('--teacher_epsilon_decay', type=float, default=0.90, help='Epsilon decay rate for teacher')
    
    # Environment and logging
    parser.add_argument('--device_id', type=int, default=6, help='GPU device ID')
    parser.add_argument('--log_dir', type=str, default='logs/autocalc_logs', help='Log directory')
    parser.add_argument('--use_wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--validation_episodes', type=int, default=10, help='Episodes per validation environment')
    
    # Evaluation
    parser.add_argument('--eval', action='store_true', help='Run evaluation after training')
    parser.add_argument('--eval_only', action='store_true', help='Run evaluation only, skipping training')
    parser.add_argument('--eval_episodes', type=int, default=10, help='Number of episodes for evaluation')
    parser.add_argument('--eval_model_path', type=str, default=None, 
                       help='Path to model for evaluation. If not provided, uses best_student_model.zip if available, otherwise final_student_model.zip')
    parser.add_argument('--max_episode_length', type=int, default=250, help='Maximum episode length for evaluation')
    
    return parser.parse_args()

def evaluate_student_model(args, log_dir, model_path=None, task_name=None, seed=None, 
                       max_episode_length=250, skip_frame=3, num_episodes=10):
    """Evaluate the student model using the causal world benchmark"""
    import json
    
    task_name = task_name or args.task
    seed = seed or args.seed
    
    set_seed(seed)
    logging.info(f"Running comprehensive evaluation on task: {task_name}")
    
    # Create the environment
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

    # Determine which model to evaluate
    if model_path is None:
        # Default to the final model
        model_path = os.path.join(log_dir, "final_student_model.zip")
        # Check if we should use the best model instead
        best_model_path = os.path.join(log_dir, "best_student_model.zip")
        if os.path.exists(best_model_path):
            model_path = best_model_path
            logging.info("Using best model for evaluation based on validation performance")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Student model not found at {model_path}")
    
    # Load the model
    model = PPO.load(model_path)
    logging.info(f"Loaded model from {model_path}")

    # Basic episode evaluation
    logging.info("\nRunning episode-based evaluation:")
    all_rewards, all_successes, episode_lengths = [], [], []
    
    for ep in range(num_episodes):
        obs = env.reset()
        if hasattr(env, 'seed'):
            env.seed(seed + ep)
        
        done = False
        total_reward, successes, steps = 0.0, 0, 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            if isinstance(info, dict) and 'success' in info:
                successes += int(info['success'])
        
        logging.info(f"Episode {ep + 1}: reward = {total_reward:.2f}, success = {successes}, steps = {steps}")
        all_rewards.append(total_reward)
        all_successes.append(successes > 0)  # Count episodes with at least one success
        episode_lengths.append(steps)
    
    # Aggregate metrics
    mean_reward = np.mean(all_rewards)
    success_rate = np.mean(all_successes)
    mean_episode_length = np.mean(episode_lengths)
    
    logging.info(f"\nEvaluation Results:")
    logging.info(f"Mean reward: {mean_reward:.4f}")
    logging.info(f"Success rate: {success_rate:.4f}")
    logging.info(f"Mean episode length: {mean_episode_length:.2f}")

    # Run benchmark evaluation
    logging.info("\nRunning benchmark evaluation...")
    if task_name not in TASK_BENCHMARKS:
        logging.error(f"No benchmark available for task: '{task_name}'. Supported: {SUPPORTED_TASKS}")
        return {
            'mean_reward': mean_reward,
            'success_rate': success_rate,
            'mean_episode_length': mean_episode_length
        }
    
    benchmark = TASK_BENCHMARKS[task_name]
    
    # Use the causalworld benchmark evaluation
    evaluation = EvaluationPipeline(
        evaluation_protocols=benchmark['evaluation_protocols'],
        task_params={'task_generator_id': task_name},
        world_params={'skip_frame': skip_frame, 'action_mode': 'joint_torques'},
        visualize_evaluation=False
    )

    # Create policy function
    def policy_fn(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action
    
    # Run benchmark evaluation
    scores = evaluation.evaluate_policy(policy_fn, fraction=0.05)
    
    # Save benchmark results
    benchmark_results = {
        'mean_reward': float(mean_reward),
        'success_rate': float(success_rate),
        'mean_episode_length': float(mean_episode_length),
        'benchmark_scores': scores
    }
    
    benchmark_path = os.path.join(log_dir, "evaluation_results.json")
    with open(benchmark_path, 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    logging.info(f"Evaluation results saved to: {benchmark_path}")
    
    # Generate visualizations
    plots_dir = os.path.join(log_dir, "evaluation_plots")
    os.makedirs(plots_dir, exist_ok=True)
    try:
        vis.generate_visual_analysis(plots_dir, experiments={task_name: scores})
        logging.info(f"Visualization saved to: {plots_dir}")
    except Exception as e:
        logging.error(f"Error generating visualizations: {e}")
    
    env.close()
    return benchmark_results

def main():
    """Main training function"""
    args = parse_args()
    set_seed(args.seed)

    if args.eval_only:
        logging.info("Running in evaluation-only mode.")
        if not os.path.exists(args.log_dir):
            logging.error(f"Log directory not found for evaluation: {args.log_dir}")
            return
        
        # Ensure we use the best model for eval_only
        model_path = os.path.join(args.log_dir, "best_student_model.zip")
        if not os.path.exists(model_path):
            logging.error(f"Best student model not found in {args.log_dir}")
            return

        evaluate_student_model(
            args=args,
            log_dir=args.log_dir,
            model_path=model_path,
            task_name=args.task,
            seed=args.seed,
            max_episode_length=args.max_episode_length,
            num_episodes=args.eval_episodes
        )
        logging.info("Evaluation complete.")
        return
    
    # Initialize AutoCaLC framework
    autocalc = AutoCaLC(args)
    
    # Train the models
    student, teacher = autocalc.train()
    
    # Optionally run evaluation
    if args.eval:
        logging.info("Running final evaluation...")
        evaluate_student_model(
            args=args,
            log_dir=args.log_dir,
            model_path=args.eval_model_path,
            task_name=args.task,
            seed=args.seed,
            max_episode_length=args.max_episode_length,
            num_episodes=args.eval_episodes
        )
    
    logging.info("AutoCaLC training completed successfully!")

if __name__ == "__main__":
    main()