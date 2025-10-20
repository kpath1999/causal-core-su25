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

TRAINING COMMANDS:

Basic training with evaluation:
python meta_teacher_student_qtable.py --task pushing --meta_episodes 50 --student_train_steps 50000 --alpha 0.1 --beta 1.0 --gamma 0.9 --eval --use_wandb

Quick test run (fast debugging):
python meta_teacher_student_qtable.py --task pushing --meta_episodes 5 --student_train_steps 1000 --validation_episodes 3

Full training with custom parameters:
python meta_teacher_student_qtable.py --task pushing --meta_episodes 50 --student_train_steps 5000 \
    --alpha 0.1 --gamma 0.9 --beta 1.0 \
    --validation_episodes 10 --eval --use_wandb --device_id 6

EVALUATION COMMANDS:

Evaluate trained model (uses best model if available):
python meta_teacher_student_qtable.py --eval_only --log_dir logs/autocalc_qtable --task pushing --eval_episodes 20

Evaluate with custom model path:
python meta_teacher_student_qtable.py --eval_only --eval_model_path logs/my_experiment/final_student_model.zip \
    --task pushing --eval_episodes 20 --max_episode_length 300

SUPPORTED TASKS: pushing, reaching, picking, pick_and_place, stacking2

For all options: python meta_teacher_student_qtable.py --help
"""

# NOTE: it would be cool to see the Q-table created as a 7x7 grid where I can see the Q-vals evolve across the 50 meta-episodes

import numpy as np
import os
import json
import argparse
import logging
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from copy import deepcopy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.utils import set_random_seed

# importing components I already created in baselines.py
from baselines import (
    INTERVENTIONS,
    TASK_BENCHMARKS,
    SUPPORTED_TASKS,
    DENSE_REWARD_WEIGHTS,
    create_environment,
    RewardMonitorCallback,
    CSVLogger,
    ValidationCallback
)

# Additional imports for evaluation
from causal_world.task_generators import generate_task
from causal_world.envs import CausalWorld
from causal_world.evaluation import EvaluationPipeline
try:
    from causal_world.evaluation import visualize_evaluation as vis
    VIS_AVAILABLE = True
except ImportError:
    VIS_AVAILABLE = False
    logging.warning("Visualization module not available")

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
        self.visit_counts = np.zeros(num_interventions)     # 1D array for interventions

        # state tracking
        self.current_state = None

        # training statistics
        self.training_stats = {
            'q_updates': 0,
            'exploration_bonuses': [],
            'selected_actions': [],
            'selected_indices': [],
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
            if self.visit_counts[action_idx] == 0:
                exploration_bonus = float('inf')    # forcing exploration of unvisited actions
            else:
                exploration_bonus = self.beta / np.sqrt(self.visit_counts[action_idx])
            
            ucb_values[action_idx] = q_value + exploration_bonus
        
        # select action with the highest UCB value
        action_idx = int(np.argmax(ucb_values))

        # track statistics
        self.training_stats['exploration_bonuses'].append(float(ucb_values[action_idx] - self.q_table[state_idx, action_idx]))
        self.training_stats['selected_actions'].append(f"{INTERVENTION_NAMES[state_idx]}_to_{INTERVENTION_NAMES[action_idx]}")
        self.training_stats['selected_indices'].append(action_idx)
        self.training_stats['ucb_values'].append(ucb_values.tolist())
        
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
        self.visit_counts[action_idx] += 1
        self.training_stats['q_updates'] += 1
        self.training_stats['meta_rewards'].append(float(reward))

        # return loss equivalent for logging
        return abs(td_error)
    
    def get_optimal_sequence(self, start_state_idx, max_length=50):
        """extract optimal intervention sequence from the learned Q-table"""
        current_state = int(start_state_idx)
        sequence = [current_state]

        for _ in range(max_length):
            best_action = int(np.argmax(self.q_table[current_state, :]))
            sequence.append(best_action)
            current_state = best_action

            # OPTIONAL: cycle detection mechanism that doesn't break but warns
            # this would detect patterns like A->B->C->A->B->C without stopping

        return sequence
    
    def get_training_stats(self):
        """returning the training statistics"""
        if not self.training_stats['meta_rewards']:
            return {}
        
        return {
            'avg_meta_reward': float(np.mean(self.training_stats['meta_rewards'][-50:])),
            'avg_exploration_bonus': float(np.mean(self.training_stats['exploration_bonuses'][-50:])),
            'total_q_updates': int(self.training_stats['q_updates']),
            'q_table_max': float(np.max(self.q_table)),
            'q_table_min': float(np.min(self.q_table)),
            'unvisited_actions': int(np.sum(self.visit_counts == 0))
        }

    def save(self, path):
        """save Q-table and stats"""
        save_data = {
            'q_table': self.q_table.tolist(),
            'visit_counts': [int(v) for v in self.visit_counts.tolist()],
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
    
    intervention_type = last_intervention['type'] if hasattr(last_intervention, 'type') else 'random'
    return INTERVENTION_TO_IDX.get(intervention_type, 0)

class AutoCaLC:
    """main autoalc framework with a tabular Q-learning teacher"""
    # using discrete states and simple Q-table updates
    def __init__(self, args):
        self.args = args

        os.makedirs(self.args.log_dir, exist_ok=True)

        # initialize the student
        self.student = self._create_student()

        # initialize tabular teacher
        self.teacher = Teacher(
            num_interventions=len(INTERVENTIONS),
            alpha=args.alpha,
            gamma=args.gamma,
            beta=args.beta
        )

        # tracking structures for analysis and logging
        self.curriculum_history = []
        self.state_history = []
        self.meta_rewards_history = []
        self.validation_history = []
        self.meta_step_records = []
        self.q_table_history = [self.teacher.q_table.copy()]
        self.snapshot_dir = os.path.join(self.args.log_dir, "qtable_snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.initial_validation_reward = None

        # initialize the logging
        self.csvlogger = CSVLogger(log_dir=args.log_dir)

        # initialize wandb if available
        if args.use_wandb and WANDB_AVAILABLE:
            self._setup_wandb()
    
    def _create_student(self):
        """create PPO student agent by loading pretrained model"""
        # determine pretrained model path
        if hasattr(self.args, 'pretrained_path') and self.args.pretrained_path is not None:
            pretrained_path = self.args.pretrained_path
        else:
            pretrained_path = f'models/ppo_{self.args.task}_sb3/final_model.zip'
        
        # load pretrained model
        set_random_seed(self.args.student_seed)
        # kausar (oct 20): adding clipping and entropy to prevent NaNs; max_grad_norm is the best strategy
        student = PPO.load(
            pretrained_path,
            learning_rate=3e-5,      # slightly lower than default 3e-4
            ent_coef=0.01,           # regularizes entropy; helps avoid log(0) NaNs
            max_grad_norm=0.5,       # prevents gradient explosion
            clip_range=0.2,          # ensures stable policy updates
            clip_range_vf=0.2,       # prevents critic over-updates (major NaN source)
            vf_coef=0.5,             # keep balanced critic loss
            normalize_advantage=True # ensures stable advantage magnitudes
        )

        logging.info(f"Loaded pretrained student model from: {pretrained_path}")
        
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
    
    def _train_student_on_intervention(self, intervention, meta_step, cumulative_timesteps):
        """train student on selected intervention with proper logging"""
        # create env with intervention
        env = create_environment(self.args.task, intervention, seed=self.args.student_seed + meta_step)
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecMonitor(vec_env)

        # set env for student
        self.student.set_env(vec_env)

        # Set up callbacks for logging
        reward_monitor = RewardMonitorCallback(
            intervention_type=intervention['type'] if intervention else 'base',
            csv_logger=self.csvlogger,
            stage=meta_step,
            cumulative_timesteps=cumulative_timesteps,
            baseline_type="autocalc_qtable"
        )
        
        callbacks = [reward_monitor]
        
        # Add wandb callback if enabled
        if self.args.use_wandb and WANDB_AVAILABLE:
            wandb_callback = WandbCallback(
                gradient_save_freq=100,
                model_save_path=self.args.log_dir,
                verbose=0
            )
            callbacks.append(wandb_callback)
        
        callback_list = CallbackList(callbacks)

        # train student
        logging.info(f"Training student on {intervention['type'] if intervention else 'base'} for {self.args.student_train_steps} steps")
        self.student.learn(
            total_timesteps=self.args.student_train_steps,
            callback=callback_list,
            reset_num_timesteps=False
        )

        env.close()
    
    def _run_validation(self, meta_step, cumulative_timesteps):
        """run validation protocol with proper logging"""
        logging.info(f"Running validation after meta-episode {meta_step}")
        
        # Create validation callback
        validation_callback = ValidationCallback(
            validation_frequency=float('inf'),  # only run when explicitly called
            task_name=self.args.task,
            csv_logger=self.csvlogger,
            stage=meta_step,
            cumulative_timesteps=cumulative_timesteps,
            validation_episodes=self.args.validation_episodes,
            seed=1000,  # fixed validation seed
            baseline_type="autocalc_qtable"
        )
        
        # Run validation and get metrics
        validation_metrics = validation_callback._execute_validation(self.student)
        
        return validation_metrics['validation_avg_reward']
    
    def _log_qtable_progress(self, meta_step, state_idx, action_idx, meta_reward):
        """Enhanced Q-table logging (replaces neural network logging)"""
        logging.info(f"Q-TABLE UPDATE [Step {meta_step}]:")
        logging.info(f"  State: {INTERVENTION_NAMES[state_idx]}")
        logging.info(f"  Action: {INTERVENTION_NAMES[action_idx]}")
        logging.info(f"  Meta-reward: {meta_reward:.4f}")
        logging.info(f"  New Q-value: {self.teacher.q_table[state_idx, action_idx]:.4f}")
        logging.info(f"  Visit count: {int(self.teacher.visit_counts[action_idx])}")
        
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
                'visit_count': self.teacher.visit_counts[action_idx],
                'exploration_bonus': self.teacher.training_stats['exploration_bonuses'][-1] if self.teacher.training_stats['exploration_bonuses'] else 0,
                **{f'q_table_{i}_{j}': self.teacher.q_table[i,j] for i in range(7) for j in range(7)}
            })

    def _save_qtable_snapshot(self, meta_step):
        """Persist a heatmap/table view of the current Q-table."""
        filename = os.path.join(self.snapshot_dir, f"qtable_{meta_step:03d}.png")
        q_values = self.teacher.q_table

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(f"Q-table After Meta-Step {meta_step}")

        vmin = float(np.min(q_values))
        vmax = float(np.max(q_values))
        if np.isclose(vmin, vmax):
            vmin -= 1e-6
            vmax += 1e-6

        heatmap = ax.imshow(q_values, cmap='coolwarm', vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(INTERVENTION_NAMES)))
        ax.set_yticks(range(len(INTERVENTION_NAMES)))
        ax.set_xticklabels(INTERVENTION_NAMES, rotation=45, ha='right')
        ax.set_yticklabels(INTERVENTION_NAMES)

        for i in range(len(INTERVENTION_NAMES)):
            for j in range(len(INTERVENTION_NAMES)):
                ax.text(j, i, f"{q_values[i, j]:.2f}", ha='center', va='center', color='black')

        fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04, label='Q-value')
        plt.tight_layout()
        fig.savefig(filename, dpi=200)
        plt.close(fig)
        logging.info(f"  Saved Q-table snapshot to {os.path.relpath(filename, self.args.log_dir)}")
    
    def train(self):
        """main training loop with tabular Q-learning"""
        logging.info("Starting AutoCaLC training with tabular Q-learning teacher")

        # initialize with random starting intervention
        last_intervention_idx = 0       # start with 'goal'
        self.curriculum_history = []
        self.meta_rewards_history = []
        self.validation_history = []
        self.meta_step_records = []
        self.q_table_history = [self.teacher.q_table.copy()]
        self.state_history = [int(last_intervention_idx)]
        
        # track cumulative timesteps for logging
        cumulative_timesteps = 0

        # get initial validation performance
        logging.info("Computing initial validation performance...")
        last_validation_reward = self._run_validation(0, cumulative_timesteps)
        logging.info(f"Initial validation reward: {last_validation_reward:.4f}")
        self.initial_validation_reward = float(last_validation_reward)
        self.validation_history = [float(last_validation_reward)]
        self._save_qtable_snapshot(0)

        # main meta-learning loop
        for meta_step in range(1, self.args.meta_episodes + 1):
            logging.info("="*80)
            logging.info(f"Meta-Episode {meta_step}/{self.args.meta_episodes}")

            # 1. current state = last intervention used (discrete)
            current_state_idx = last_intervention_idx

            # 2. teacher selects next intervention using Q-table + UCB
            selected_intervention_idx = self.teacher.select_action(current_state_idx, meta_step)
            selected_intervention = INTERVENTIONS[selected_intervention_idx]
            self.curriculum_history.append(int(selected_intervention_idx))

            logging.info(f"STATE: {INTERVENTION_NAMES[current_state_idx]} -> ACTION: {INTERVENTION_NAMES[selected_intervention_idx]}")

            # 3. train student on selected intervention
            self._train_student_on_intervention(selected_intervention, meta_step, cumulative_timesteps)
            cumulative_timesteps += self.args.student_train_steps

            # 4. validate and compute meta-reward
            current_validation_reward = self._run_validation(meta_step, cumulative_timesteps)
            meta_reward = current_validation_reward - last_validation_reward    # this is the learning progress
            self.meta_rewards_history.append(float(meta_reward))
            self.validation_history.append(float(current_validation_reward))

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
            self._save_qtable_snapshot(meta_step)

            self.q_table_history.append(self.teacher.q_table.copy())
            self.state_history.append(int(next_state_idx))
            self.meta_step_records.append({
                'meta_step': int(meta_step),
                'state_idx': int(current_state_idx),
                'state_name': INTERVENTION_NAMES[current_state_idx],
                'action_idx': int(selected_intervention_idx),
                'action_name': INTERVENTION_NAMES[selected_intervention_idx],
                'validation_reward': float(current_validation_reward),
                'meta_reward': float(meta_reward),
                'td_error': float(td_error),
                'q_value_post_update': float(self.teacher.q_table[current_state_idx, selected_intervention_idx]),
                'visit_count': int(self.teacher.visit_counts[selected_intervention_idx])
            })

            # 7. update for next iteration
            last_intervention_idx = selected_intervention_idx
            last_validation_reward = current_validation_reward

            # 8. save checkpoint every 10 episodes
            if meta_step % 10 == 0:
                checkpoint_path = os.path.join(self.args.log_dir, f"qtable_checkpoint_{meta_step}.json")
                self.teacher.save(checkpoint_path)
                
                # Also save student model checkpoints
                student_checkpoint_path = os.path.join(self.args.log_dir, f"student_model_step_{meta_step}.zip")
                self.student.save(student_checkpoint_path)
                logging.info(f"Saved student checkpoint to {student_checkpoint_path}")
        
        # training is now complete - I can extract and save the learned curriculum
        logging.info("Training complete! Extracting learned curriculum...")
        optimal_sequence = self._extract_learned_curriculum()
        
        # Save final student model
        final_student_path = os.path.join(self.args.log_dir, "final_student_model.zip")
        self.student.save(final_student_path)
        logging.info(f"Final student model saved to {final_student_path}")
        
        return optimal_sequence

    def _extract_learned_curriculum(self):
        """Extract optimal curriculum from learned Q-table"""
        executed_indices = [int(idx) for idx in self.curriculum_history]
        executed_names = [INTERVENTION_NAMES[idx] for idx in executed_indices]
        executed_str = " -> ".join(executed_names) if executed_names else "(no interventions selected)"

        greedy_full = [int(idx) for idx in self.teacher.get_optimal_sequence(0, max_length=self.args.meta_episodes)]
        greedy_actions = greedy_full[1:] if len(greedy_full) > 1 else []
        greedy_action_names = [INTERVENTION_NAMES[idx] for idx in greedy_actions]
        greedy_str = " -> ".join(greedy_action_names) if greedy_action_names else "(no actions)"

        logging.info("="*80)
        logging.info("EXECUTED CURRICULUM (actions taken during training):")
        logging.info(executed_str)
        logging.info("-"*80)
        logging.info("GREEDY CURRICULUM FROM FINAL Q-TABLE:")
        logging.info(greedy_str)
        logging.info("="*80)
        
        # Save curriculum to file
        curriculum_path = os.path.join(self.args.log_dir, "learned_curriculum.json")
        stats_summary = self.teacher.get_training_stats()
        snapshot_files = [os.path.join('qtable_snapshots', fname) for fname in sorted(os.listdir(self.snapshot_dir))]
        curriculum_data = {
            'executed_curriculum': {
                'action_indices': executed_indices,
                'action_names': executed_names,
                'state_indices': [int(idx) for idx in self.state_history],
                'state_names': [INTERVENTION_NAMES[idx] for idx in self.state_history],
                'sequence_string': executed_str
            },
            'greedy_curriculum': {
                'full_sequence_indices': greedy_full,
                'full_sequence_names': [INTERVENTION_NAMES[idx] for idx in greedy_full],
                'action_indices': greedy_actions,
                'action_names': greedy_action_names,
                'sequence_string': greedy_str
            },
            'initial_validation_reward': self.initial_validation_reward,
            'validation_rewards': [float(v) for v in self.validation_history],
            'meta_rewards': [float(r) for r in self.meta_rewards_history],
            'meta_step_records': self.meta_step_records,
            'final_q_table': self.teacher.q_table.tolist(),
            'q_table_history': [q.tolist() for q in self.q_table_history],
            'visit_counts': [int(v) for v in self.teacher.visit_counts.tolist()],
            'training_stats_summary': stats_summary,
            'qtable_snapshot_dir': self.snapshot_dir,
            'qtable_snapshot_files': snapshot_files
        }
        
        with open(curriculum_path, 'w') as f:
            json.dump(curriculum_data, f, indent=2)
        
        logging.info(f"Curriculum saved to {curriculum_path}")
        return executed_indices

def evaluate_student_model(args, log_dir, model_path=None, task_name=None, seed=None, 
                       max_episode_length=250, skip_frame=3, num_episodes=10):
    """Evaluate the student model using the causal world benchmark"""
    import json
    
    task_name = task_name or args.task
    seed = seed or args.student_seed
    
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
            logging.info(f"Using best model instead of final model")
    
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
        total_reward, successes, steps = 0, 0, 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if info.get('success', False):
                successes += 1
            steps += 1
        
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
        benchmark_results = {
            'mean_reward': float(mean_reward),
            'success_rate': float(success_rate),
            'mean_episode_length': float(mean_episode_length),
            'benchmark_scores': None
        }
    else:
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
    
    # Generate visualizations if available
    if benchmark_results['benchmark_scores'] is not None and VIS_AVAILABLE:
        plots_dir = os.path.join(log_dir, "evaluation_plots")
        os.makedirs(plots_dir, exist_ok=True)
        try:
            vis.generate_visual_analysis(plots_dir, experiments={task_name: benchmark_results['benchmark_scores']})
            logging.info(f"Visualization saved to: {plots_dir}")
        except Exception as e:
            logging.error(f"Error generating visualizations: {e}")
    
    env.close()
    return benchmark_results

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
    parser.add_argument('--device_id', type=int, default=6, help='GPU device ID')
    
    # Saving and loading
    parser.add_argument('--save_qtable', type=str, default=None, help='Path to save final Q-table')
    parser.add_argument('--load_qtable', type=str, default=None, help='Path to load initial Q-table')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained student model')
    parser.add_argument('--eval', action='store_true', help='Run evaluation after training')
    
    # Evaluation parameters
    parser.add_argument('--eval_only', action='store_true', help='Run evaluation only, skipping training')
    parser.add_argument('--eval_episodes', type=int, default=10, help='Number of episodes for evaluation')
    parser.add_argument('--eval_model_path', type=str, default=None, 
                       help='Path to model for evaluation. If not provided, uses best_student_model.zip if available, otherwise final_student_model.zip')
    parser.add_argument('--max_episode_length', type=int, default=250, help='Maximum episode length for evaluation')
    parser.add_argument('--skip_frame', type=int, default=3, help='Frame skip for environment')
    
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
    
    # Handle evaluation-only mode
    if args.eval_only:
        logging.info("Running in evaluation-only mode.")
        # Determine model path
        if args.eval_model_path:
            model_path = args.eval_model_path
        else:
            # Check for best model, fall back to final model
            best_model_path = os.path.join(args.log_dir, "best_student_model.zip")
            final_model_path = os.path.join(args.log_dir, "final_student_model.zip")
            
            if os.path.exists(best_model_path):
                model_path = best_model_path
                logging.info(f"Using best student model for evaluation")
            elif os.path.exists(final_model_path):
                model_path = final_model_path
                logging.info(f"Using final student model for evaluation")
            else:
                logging.error(f"No model found in {args.log_dir}")
                return
        
        # Run evaluation
        eval_results = evaluate_student_model(
            args=args,
            log_dir=args.log_dir,
            model_path=model_path,
            task_name=args.task,
            seed=args.student_seed,
            max_episode_length=args.max_episode_length,
            skip_frame=args.skip_frame,
            num_episodes=args.eval_episodes
        )
        logging.info("Evaluation completed successfully!")
        return
    
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
        
        # First run simple validation
        final_validation = autocalc._run_validation(args.meta_episodes)
        logging.info(f"Final validation performance: {final_validation:.4f}")
        
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({'final_validation': final_validation})
        
        # Run comprehensive benchmark evaluation
        try:
            eval_results = evaluate_student_model(
                args=args,
                log_dir=args.log_dir,
                model_path=None,  # Will use final_student_model.zip
                task_name=args.task,
                seed=args.student_seed,
                max_episode_length=args.max_episode_length,
                skip_frame=args.skip_frame,
                num_episodes=args.eval_episodes
            )
            logging.info("Benchmark evaluation completed successfully!")
            
            if args.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'eval/mean_reward': eval_results['mean_reward'],
                    'eval/success_rate': eval_results['success_rate'],
                    'eval/mean_episode_length': eval_results['mean_episode_length']
                })
        except Exception as e:
            logging.error(f"Error during benchmark evaluation: {e}")
            import traceback
            traceback.print_exc()
        
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.finish()
    
    logging.info("AutoCaLC training completed successfully!")
    return optimal_sequence

if __name__ == "__main__":
    main()