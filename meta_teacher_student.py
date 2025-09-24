"""
AutoCaLC (meta-teacher/student) runner.

Summary
- Iterate over M meta-episodes:
  1) compute meta-state from the student,
  2) teacher picks an intervention,
  3) student trains K steps on that intervention,
  4) evaluate on a fixed validation env,
  5) meta-reward = delta(validation), update teacher.

Quick start
- Basic training
python meta_teacher_student.py --task pushing --meta_episodes 50 --student_train_steps 5000 --eval --use_wandb --device_id 7

- Advanced configuration with wandb
python meta_teacher_student.py --task pushing --meta_episodes 100 \
    --teacher_lr 1e-4 --teacher_epsilon 0.2 --sequence_length 10 \
    --use_wandb --device_id 0 --log_dir logs/experiment_1

- Quick test run
python meta_teacher_student.py --task pushing --meta_episodes 5 --student_train_steps 1000 --eval

Tips
- Multi-seed: add --teacher_seed X --student_seed Y
- Optional: --use_wandb, --device_id N
- For full options: python meta_teacher_student.py --help

Planned teacher upgrades (compact)
- Recurrent teacher state: replace MLP with LSTM/GRU so the teacher has memory over meta-episodes; keep hidden state across steps and train on trajectory subsequences from replay.
- Better meta-reward (DONE): use delta validation (r_t = val_t − val_{t-1}) to reward learning progress, not absolute level.
- Auxiliary prediction head: add a decoder to predict next meta-state; train with total_loss = q_loss + β · mse(pred_next_state, next_state).
- Double Dueling DQN: use Double-Q target (online argmax, target eval) and dueling heads (V(s) + A(s,a)) for stability and sample efficiency.
"""

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
    RewardMonitorCallback,
    IntervenedCausalWorld,
    CSVLogger,
    ValidationCallback
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
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, lstm_layers, batch_first=True, dropout=0.1)

        # Q-value heads with dueling architecture
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.advantage_head = nn.Sequential(
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
        
        # Dueling DQN: V(s) + A(s,a) - mean(A(s,a))
        value = self.value_head(last_output)
        advantage = self.advantage_head(last_output)
        
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values, new_hidden

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

class RecurrentTeacherAgent:
    """Teacher DQN agent with LSTM memory and logging"""
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, device='cpu',
                 buffer_size=10000, batch_size=32, sequence_length=5, target_update_freq=100):
        self.device = device
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.update_count = 0

        # Networks with Double DQN architecture
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
        adaptive_epsilon = max(0.05, epsilon * (0.995 ** meta_step))
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

        # Double DQN target calculation
        with torch.no_grad():
            # Use online network to select actions
            next_q_online, _ = self.q_net(next_state_seqs)
            next_actions = next_q_online.argmax(1)
            
            # Use target network to evaluate actions
            next_q_target, _ = self.target_net(next_state_seqs)
            next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            target_q = rewards + self.gamma * next_q_values
        
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

# get the teacher state
def get_teacher_state(student_model, task_name, interventions, device='cpu', seed=0):
    """
    Compute the meta-state for the teacher by evaluating reward and CM score for each intervention
    Returns interleaved array: [reward1, cm1, reward2, cm2, ..., reward7, cm7]
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
            num_episodes=5,
            seed=seed + i
        )
        rewards.append(reward_metrics['avg_reward'])

        # 2. Compute CM score
        env = create_environment(task_name, intervention, seed=seed + i + 100)
        cm_score = evaluate_cm_score(
            env, 
            student_model, 
            max_episodes=5, 
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

    # interleave rewards and CM scores
    state = np.zeros(len(INTERVENTIONS) * 2, dtype=np.float32)
    state[0::2] = normalized_rewards    # even indices for rewards
    state[1::2] = normalized_cm         # odd indices for cm scores

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
        validation_episodes=10,
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
        state_dim = len(INTERVENTIONS) * 2  # rewards + CM scores
        action_dim = len(INTERVENTIONS)
        self.teacher = RecurrentTeacherAgent(
            state_dim, 
            action_dim, 
            device=self.device,
            lr=self.args.teacher_lr,
            gamma=self.args.teacher_gamma,
            buffer_size=self.args.teacher_buffer_size,
            batch_size=self.args.teacher_batch_size,
            sequence_length=self.args.sequence_length
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

    def log_training_progress(self, meta_step, meta_reward, validation_metrics, teacher_stats, selected_intervention):
        """Comprehensive logging of training progress"""
        # Update training statistics
        self.training_stats['meta_rewards'].append(meta_reward)
        self.training_stats['validation_rewards'].append(validation_metrics['validation_avg_reward'])
        self.training_stats['validation_success_rates'].append(validation_metrics['validation_success_rate'])
        self.training_stats['selected_interventions'].append(selected_intervention['type'])
        
        if teacher_stats.get('avg_loss'):
            self.training_stats['teacher_losses'].append(teacher_stats['avg_loss'])

        # Log to console
        logging.info(f"Meta-step {meta_step}/{self.args.meta_episodes}:")
        logging.info(f"  Selected intervention: {selected_intervention['type']}")
        logging.info(f"  Meta-reward: {meta_reward:.4f}")
        logging.info(f"  Validation reward: {validation_metrics['validation_avg_reward']:.4f}")
        logging.info(f"  Validation success rate: {validation_metrics['validation_success_rate']:.3f}")
        logging.info(f"  Teacher stats: {teacher_stats}")

        # Log to wandb
        if self.args.use_wandb:
            wandb_log = {
                'meta_step': meta_step,
                'meta_reward': meta_reward,
                'validation_reward': validation_metrics['validation_avg_reward'],
                'validation_success_rate': validation_metrics['validation_success_rate'],
                'selected_intervention': selected_intervention['type'],
                **{f"teacher_{k}": v for k, v in teacher_stats.items()}
            }
            wandb.log(wandb_log)

    def train(self):
        """Main training loop with enhanced logging and adaptive teacher"""
        logging.info("Starting AutoCaLC training with recurrent teacher...")
        
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
                device=self.device, seed=self.args.seed + meta_step
            )

            # 2. Teacher selects intervention (with adaptive exploration)
            selected_intervention_idx = self.teacher.select_action(
                current_meta_state, 
                epsilon=self.args.teacher_epsilon,
                meta_step=meta_step
            )
            selected_intervention = INTERVENTIONS[selected_intervention_idx]

            logging.info(f"Teacher selected intervention: {selected_intervention['type']}")

            # 3. Train student on selected intervention
            train_env = create_environment(self.args.task, selected_intervention, seed=self.args.seed + meta_step)
            vec_env = DummyVecEnv([lambda: train_env])
            vec_env = VecMonitor(vec_env, filename=os.path.join(self.args.log_dir, f'monitor_stage{meta_step}.csv'))

            # Set up callbacks
            reward_monitor = RewardMonitorCallback(
                selected_intervention['type'], self.csv_logger, meta_step, 
                cumulative_timesteps, baseline_type="autocalc"
            )
            
            callbacks = [reward_monitor]
            if self.args.use_wandb:
                callbacks.append(WandbCallback(gradient_save_freq=1000, verbose=0))

            # Train student
            self.student.set_env(vec_env)
            self.student.learn(
                total_timesteps=self.args.student_train_steps,
                callback=CallbackList(callbacks),
                reset_num_timesteps=False
            )
            
            cumulative_timesteps += self.args.student_train_steps
            train_env.close()

            # 4. Evaluate student performance
            current_validation = run_validation_protocol(
                self.student, self.args.task, meta_step, self.args, self.csv_logger, cumulative_timesteps
            )

            # 5. Calculate meta-reward (improvement-based)
            meta_reward = current_validation['validation_avg_reward'] - last_validation_reward
            
            # 6. Update best model if needed
            self.update_best_model(current_validation['validation_avg_reward'], meta_step)

            # 7. Get next meta-state and train teacher
            next_meta_state = get_teacher_state(
                self.student, self.args.task, INTERVENTIONS,
                device=self.device, seed=self.args.seed + meta_step + 1
            )

            # Create state sequences for recurrent learning
            current_seq = list(self.teacher.state_history) if len(self.teacher.state_history) > 0 else [current_meta_state] * self.teacher.sequence_length
            next_seq = current_seq[1:] + [next_meta_state]  # Shift and add new state
            
            # Store experience and train teacher
            self.teacher.replay_buffer.push(
                np.array(current_seq), selected_intervention_idx, meta_reward, np.array(next_seq)
            )
            teacher_loss = self.teacher.train_step(meta_step)

            # 8. Get teacher statistics and log progress
            teacher_stats = self.teacher.get_training_stats()
            self.log_training_progress(
                meta_step, meta_reward, current_validation, teacher_stats, selected_intervention
            )

            # Update for next iteration
            last_validation_reward = current_validation['validation_avg_reward']

        # Save final model and statistics
        final_model_path = os.path.join(self.args.log_dir, "final_student_model.zip")
        self.student.save(final_model_path)
        
        teacher_model_path = os.path.join(self.args.log_dir, "final_teacher_model.pt")
        self.teacher.save(teacher_model_path)

        logging.info(f"Training completed!")
        logging.info(f"Best validation reward: {self.best_validation_reward:.4f} at meta-step {self.best_model_meta_step}")
        logging.info(f"Final model saved to: {final_model_path}")
        logging.info(f"Best model saved to: {self.best_model_path}")

        if self.args.use_wandb:
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
    
    # Environment and logging
    parser.add_argument('--device_id', type=int, default=6, help='GPU device ID')
    parser.add_argument('--log_dir', type=str, default='logs/autocalc_logs', help='Log directory')
    parser.add_argument('--use_wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--validation_episodes', type=int, default=10, help='Episodes per validation environment')
    
    # Evaluation
    parser.add_argument('--eval', action='store_true', help='Run evaluation after training')
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