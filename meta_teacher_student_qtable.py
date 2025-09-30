"""
AutoCaLC (meta-teacher/student) runner with a tabular Q-learning teacher.

Teacher: a simple Q-table that learns optimal intervention sequences using Upper Confidence Bound (UCB) exploration.

Steps
- Iterate over M meta-episodes:
    1) Current state = last intervention used (discrete index)
    2) Teacher selects next intervention using Q-table + UCB exploration bonus
    3) Student trains K steps on selected intervention
    4) Evaluate on fixed validation environments
    5) Meta-reward = delta(validation), update Q-table using Bellman equation

Terminal Command
python meta_teacher_student_qtable.py --task pushing --meta_episodes 50 --student_train_steps 5000 --alpha 0.1 --beta 1.0 --gamma 0.9    
"""

import numpy as np
import os
import json
import argparse
import logging
import random
from collections import defaultdict
from copy import deepcopy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

# importing components I already created in baselines.py
from baselines import (
    INTERVENTIONS,
    TASK_BENCHMARKS,
    SUPPORTED_TASKS,
    create_environment,
    RewardMonitorCallback,
    CSVLogger,
    ValidationCallback
)

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("wandb not available - logging will be CSV only")

# global constants for state/action mapping
INTERVENTION_NAMES = ['goal', 'mass', 'friction', 'visual', 'position', 'angle', 'random']
INTERVENTION_TO_IDX = {
    intervention["type"]: idx
    for idx, intervention in enumerate(INTERVENTIONS)
}

def set_seed(seed):
    """set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    logging.info(f"Random seed set to {seed}")

class Teacher:
    """tabular Q-learning teacher with UCB exploration"""
    def __init__(self, num_interventions=7, alpha=0.1, beta=1.0, gamma=0.9):
        self.num_interventions = num_interventions
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
        # data structures
        self.q_table = np.zeros((num_interventions, num_interventions))
        self.visit_counts = np.zeros(num_interventions)

        # state tracking
        self.current_state = None

        # training statistics
        self.training_stats = {
            'q_updates': 0,
            'exploration_bonuses': [],
            'selected_actions': [],
            'meta_rewards': [],
            'ucb_values': []
        }

    def select_action(self, state_idx, meta_step=0):
        """select next intervention using UCB (Upper Confidence Bound)"""
        ucb_values = np.zeros(self.num_interventions)

        for action_idx in range(self.num_interventions):
            # base q-value (exploitation)
            q_value = self.q_table[state_idx, action_idx]

            # ucb exploration bonus: beta / sqrt(N+1)
            if self.visit_counts[state_idx, action_idx] == 0:
                exploration_bonus = float('inf')    # forcing exploration of unvisited actions
            else:
                exploration_bonus = self.beta / np.sqrt(self.visit_counts[state_idx, action_idx])
            
            ucb_values[action_idx] = q_value + exploration_bonus
        
        # select action with the highest UCB value
        action_idx = np.argmax(ucb_values)

        # track statistics
        self.training_stats['exploration_bonuses'].append(ucb_values[action_idx] - self.q_table[state_idx, action_idx])
        self.training_stats['selected_actions'].append(f"{INTERVENTION_NAMES[state_idx]}_to_{INTERVENTION_NAMES[action_idx]}")
        self.training_stats['ucb_values'].append(ucb_values.copy())
        
        return action_idx
    
    def update_q_table(self, state_idx, action_idx, reward, next_state_idx):
        """update Q table using Bellman equation"""
        current_q = self.q_table[state_idx, action_idx]

        # best future Q-value
        max_next_q = np.max(self.q_table[next_state_idx, :])

        # Bellman equation: Q(s,a) = Q(s,a) + α * [r + γ * max Q(s',a') - Q(s,a)]
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q
        self.q_table[state_idx, action_idx] += self.alpha * td_error

        # update visit count
        self.visit_counts[state_idx, action_idx] += 1
        self.training_stats['q_updates'] += 1
        self.training_stats['meta_rewards'].append(reward)

        # return loss equivalent for logging
        return abs(td_error)
    
    def get_optimal_sequence(self, start_state_idx, max_length=20):
        """extract optimal intervention sequence from the learned Q-table"""
        sequence = [start_state_idx]
        current_state = start_state_idx
        visited_states = set()

        for step in range(max_length):
            # NOTE: not sure about this part
            """the issue is that we're saying no repetition is allowed, and the max len being 20 does not align with the 50 cap"""
            if current_state in visited_states:
                break   # avoid infinite loops
            visited_states.add(current_state)

            # select best action (greedy policy from Q-table)
            best_action = np.argmax(self.q_table[current_state, :])
            sequence.append(best_action)
            current_state = best_action
        
        return sequence
    
    def get_training_stats(self):
        """returning the training statistics"""
        if not self.training_stats['meta_rewards']:
            return {}
        
        return {
            'avg_meta_reward': np.mean(self.training_stats['meta_rewards'][-50:]),
            'avg_exploration_bonus': np.mean(self.training_stats['exploration_bonuses'][-50:]),
            'total_q_updates': self.training_stats['q_updates'],
            'q_table_max': np.max(self.q_table),
            'q_table_min': np.min(self.q_table),
            'unvisited_pairs': int(np.sum(self.visit_counts == 0))
        }

    def save(self, path):
        """save Q-table and stats"""
        save_data = {
            'q_table': self.q_table.tolist(),
            'visit_counts': self.visit_counts.tolist(),
            'training_stats': {k: v for k, v in self.training_stats.items() if k != 'ucb_values'},
            'hyperparams': {
                'alpha': self.alpha,
                'beta': self.beta,
                'gamma': self.gamma,
                'num_interventions': self.num_interventions
            }
        }

        with open(path, 'w') as f:
            json.dump(save_data, f, indent=2)
        logging.info(f"Q-table saved to {path}")
    
    def load(self, path):
        """load Q-table and statistics"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No Q-table found at {path}")
        
        with open(path, 'r') as f:
            save_data = json.load(f)
        
        self.q_table = np.array(save_data['q_table'])
        self.visit_counts = np.array(save_data['visit_counts'])
        self.training_stats.update(save_data.get('training_stats', {}))

        # update hyperparams if available
        if 'hyperparams' in save_data:
            hp = save_data['hyperparams']
            self.alpha = hp.get('alpha', self.alpha)
            self.beta = hp.get('beta', self.beta)
            self.gamma = hp.get('gamma', self.gamma)

        logging.info(f"Q-table loaded from {path}")


def get_current_state_idx(last_intervention):
    """convert intervention to discrete state index"""
    if last_intervention is None:
        return 0    # starting with the 'goal' intervention
    
    intervention_type = last_intervention.type if hasattr(last_intervention, 'type') else 'random'
    return INTERVENTION_TO_IDX.get(intervention_type, 0)

class AutoCaLC:
    """main autoalc framework with a tabular Q-learning teacher"""
    # using discrete states and simple Q-table updates
    def __init__(self, args):
        self.args = args

        # initialize the student
        self.student = self._create_student()

        # initialize tabular teacher
        self.teacher = Teacher(
            num_interventions=len(INTERVENTIONS),
            alpha=args.alpha,
            gamma=args.gamma,
            beta=self.beta
        )

        # initialize the logging
        self.csvlogger = CSVLogger(
            log_dir=args.log_dir,
            filename="autocalc_qtable_training.csv"
        )

        # initialize wandb if available
        if args.use_wandb and WANDB_AVAILABLE:
            self._setup_wandb()
    
    def _create_student(self):
        """create PPO student agent"""
        env = create_environment(self.args.task, None, seed=self.args.student_seed)
        vec_env = DummyVecEnv([lambda: env])

        # NOTE: should we not be loading the pretrained student model here??
        student = PPO(
            "MlpPolicy",
            vec_env,
            verbose=0,
            seed=self.args.student_seed,
            device=getattr(self.args, 'device_id', 'auto')
        )

        env.close()
        return student
    
    def _setup_wandb(self):
        """setup wandb logging (modified for Q-table)"""
        wandb.init(
            project="autocalc-qtable",
            config={
                'task': self.args.task,
                'meta_episodes': self.args.meta_episodes,
                'student_train_steps': self.args.student_train_steps,
                'alpha': self.args.alpha,
                'gamma': self.args.gamma,
                'beta': self.args.beta,
                'validation_episodes': self.args.validation_episodes
            },
            name=f"qtable_{self.args.task}_{self.args.meta_episodes}ep"
        )
    
    def _train_student_on_intervention(self, intervention, meta_step):
        """train student on selected intervention (unchanged logic from before)"""
        # create env with intervention
        env = create_environment(self.args.task, intervention, seed=self.args.student_seed + meta_step)
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecMonitor(vec_env)

        # set env for student
        self.student.set_env(vec_env)

        # train student
        logging.info(f"Training student on {intervention.type if intervention else 'base'} for {self.args.student_train_steps} steps")
        self.student.learn(total_timesteps=self.args.student_train_steps)

        env.close()
    
    def _run_validation(self, meta_step):
        """run validation protocol (unchanged from original)"""
        validation_rewards = []

        for episode in range(self.args.validation_episodes):
            # create the validation environment
            env = create_environment(
                self.args.task,
                None,
                seed=1000 + episode     # fixed val seeds
            )

            obs = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action, _ = self.student.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward
            
            validation_rewards.append(episode_reward)
            env.close()

        return np.mean(validation_rewards)
    
    def _log_qtable_progress(self, meta_step, state_idx, action_idx, meta_reward):
        """Enhanced Q-table logging (replaces neural network logging)"""
        logging.info(f"Q-TABLE UPDATE [Step {meta_step}]:")
        logging.info(f"  State: {INTERVENTION_NAMES[state_idx]}")
        logging.info(f"  Action: {INTERVENTION_NAMES[action_idx]}")
        logging.info(f"  Meta-reward: {meta_reward:.4f}")
        logging.info(f"  New Q-value: {self.teacher.q_table[state_idx, action_idx]:.4f}")
        logging.info(f"  Visit count: {int(self.teacher.visit_counts[state_idx, action_idx])}")
        
        # Log full Q-table every 10 steps
        if meta_step % 10 == 0:
            logging.info("\nCurrent Q-Table:")
            logging.info("State\\Action  " + "".join([f"{name:>8s}" for name in INTERVENTION_NAMES]))
            for i, state_name in enumerate(INTERVENTION_NAMES):
                row_str = f"{state_name:>12s}: " + "".join([f"{self.teacher.q_table[i,j]:8.3f}" for j in range(7)])
                logging.info(row_str)
        
        # Log to wandb if available
        if hasattr(self, 'args') and self.args.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'meta_step': meta_step,
                'meta_reward': meta_reward,
                'q_value': self.teacher.q_table[state_idx, action_idx],
                'visit_count': self.teacher.visit_counts[state_idx, action_idx],
                'exploration_bonus': self.teacher.training_stats['exploration_bonuses'][-1] if self.teacher.training_stats['exploration_bonuses'] else 0,
                **{f'q_table_{i}_{j}': self.teacher.q_table[i,j] for i in range(7) for j in range(7)}
            })
    
    def train(self):
        """main training loop with tabular Q-learning"""
        logging.info("Starting AutoCaLC training with tabular Q-learning teacher")

        # initialize with random starting intervention
        last_intervention_idx = 0       # start with 'goal'

        # get initial validation performance
        logging.info("Computing initial validation performance...")
        last_validation_reward = self._run_validation(0)
        logging.info(f"Initial validation reward: {last_validation_reward:.4f}")

        # main meta-learning loop
        for meta_step in range(1, self.args.meta_episodes + 1):
            logging.info("="*80)
            logging.info(f"Meta-Episode {meta_step}/{self.args.meta_episodes}")

            # 1. current state = last intervention used (discrete)
            current_state_idx = last_intervention_idx

            # 2. teacher selects next intervention using Q-table + UCB
            selected_intervention_idx = self.teacher.select_action(current_state_idx, meta_step)
            selected_intervention = INTERVENTIONS[selected_intervention_idx]

            logging.state(f"STATE: {INTERVENTION_NAMES[current_state_idx]} -> ACTION: {INTERVENTION_NAMES[selected_intervention_idx]}")

            # 3. train student on selected intervention
            self._train_student_on_intervention(selected_intervention, meta_step)

            # 4. validate and compute meta-reward
            current_validation_reward = self._run_validation(meta_step)
            meta_reward = current_validation_reward - last_validation_reward    # this is the learning progress

            # 5. update Q-table using Bellman equation
            next_state_idx = selected_intervention_idx      # next state = action taken
            td_error = self.teacher.update_q_table(
                current_state_idx,
                selected_intervention_idx,
                meta_reward,
                next_state_idx
            )

            # 6. log the progress
            self._log_qtable_progress(meta_step, current_state_idx, selected_intervention_idx, meta_reward)

            # 7. update for next iteration
            last_intervention_idx = selected_intervention_idx
            last_validation_reward = current_validation_reward

            # 8. save checkpoint every 10 episodes
            if meta_step % 10 == 0:
                checkpoint_path = os.path.join(self.args.log_dir, f"qtable_checkpoint_{meta_step}.json")
                self.teacher.save(checkpoint_path)
        
        # training is now complete - I can extract and save the learned curriculum
        logging.info("Training complete! Extracting learned curriculum...")
        optimal_sequence = self._extract_learned_curriculum()
        
        return optimal_sequence

    def _extract_learned_curriculum(self):
        """Extract optimal curriculum from learned Q-table"""
        optimal_sequence = self.teacher.get_optimal_sequence(0)  # Start from 'goal'
        
        logging.info("="*80)
        logging.info("LEARNED OPTIMAL CURRICULUM:")
        curriculum_str = " -> ".join([INTERVENTION_NAMES[idx] for idx in optimal_sequence])
        logging.info(curriculum_str)
        logging.info("="*80)
        
        # Save curriculum to file
        curriculum_path = os.path.join(self.args.log_dir, "learned_curriculum.json")
        curriculum_data = {
            'sequence_indices': optimal_sequence,
            'sequence_names': [INTERVENTION_NAMES[idx] for idx in optimal_sequence],
            'curriculum_string': curriculum_str,
            'final_q_table': self.teacher.q_table.tolist(),
            'visit_counts': self.teacher.visit_counts.tolist(),
            'training_stats': self.teacher.get_training_stats()
        }
        
        with open(curriculum_path, 'w') as f:
            json.dump(curriculum_data, f, indent=2)
        
        logging.info(f"Curriculum saved to {curriculum_path}")
        return optimal_sequence

def parse_args():
    """Parse command line arguments (modified for Q-table hyperparameters)"""
    parser = argparse.ArgumentParser(description="AutoCaLC with Tabular Q-Learning Teacher")
    
    # Task and training
    parser.add_argument('--task', type=str, default='pushing', choices=SUPPORTED_TASKS)
    parser.add_argument('--meta_episodes', type=int, default=50, help='Number of meta-episodes')
    parser.add_argument('--student_train_steps', type=int, default=5000, help='Student training steps per meta-episode')
    parser.add_argument('--validation_episodes', type=int, default=10, help='Episodes for validation')
    
    # Q-learning hyperparameters (replace neural network params)
    parser.add_argument('--alpha', type=float, default=0.1, help='Q-learning rate')
    parser.add_argument('--gamma', type=float, default=0.9, help='Discount factor')
    parser.add_argument('--beta', type=float, default=1.0, help='UCB exploration coefficient')
    
    # Seeds and logging
    parser.add_argument('--student_seed', type=int, default=42, help='Student random seed')
    parser.add_argument('--teacher_seed', type=int, default=123, help='Teacher random seed')
    parser.add_argument('--log_dir', type=str, default='logs/autocalc_qtable', help='Log directory')
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb logging')
    parser.add_argument('--device_id', type=str, default='auto', help='Device for training')
    
    # Saving and loading
    parser.add_argument('--save_qtable', type=str, default=None, help='Path to save final Q-table')
    parser.add_argument('--load_qtable', type=str, default=None, help='Path to load initial Q-table')
    parser.add_argument('--eval', action='store_true', help='Run evaluation after training')
    
    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_args()
    
    # Setup logging
    os.makedirs(args.log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    # Set seeds
    set_seed(args.teacher_seed)
    
    # Create AutoCaLC framework
    logging.info(f"Initializing AutoCaLC with tabular Q-learning teacher...")
    logging.info(f"Task: {args.task}")
    logging.info(f"Meta-episodes: {args.meta_episodes}")
    logging.info(f"Q-learning hyperparameters: α={args.alpha}, γ={args.gamma}, β={args.beta}")
    
    autocalc = AutoCaLC(args)
    
    # Load pre-trained Q-table if specified
    if args.load_qtable:
        autocalc.teacher.load(args.load_qtable)
    
    # Train the system
    optimal_sequence = autocalc.train()
    
    # Save final Q-table
    if args.save_qtable:
        autocalc.teacher.save(args.save_qtable)
    else:
        default_save_path = os.path.join(args.log_dir, "final_qtable.json")
        autocalc.teacher.save(default_save_path)
    
    # Evaluation
    if args.eval:
        logging.info("Running final evaluation...")
        final_validation = autocalc._run_validation(args.meta_episodes)
        logging.info(f"Final validation performance: {final_validation:.4f}")
        
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({'final_validation': final_validation})
            wandb.finish()
    
    logging.info("AutoCaLC training completed successfully!")
    return optimal_sequence

if __name__ == "__main__":
    main()