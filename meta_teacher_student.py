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
- Train + evaluate:
  python meta_teacher_student.py --task pushing --student_train_steps 50000 --meta_episodes 50 --eval --log_dir logs/autocalc_logs

- Pretrain teacher (examples: 100k/175k/250k):
  python meta_teacher_student.py --task pushing --teacher_pretrain_steps 100000 --teacher_pretrain_only --save_pretrained_teacher models/teacher_100k.pt --log_dir logs/teacher_pretrain_100k

- Use a pretrained teacher (example: 100k):
  python meta_teacher_student.py --task pushing --load_pretrained_teacher models/teacher_100k.pt --log_dir logs/autocalc_teacher100k

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


class AutoCaLC:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. initialize student and teacher
        self.student = self._initialize_student()
        self.teacher = TeacherDQNAgent(
            statedim=len(INTERVENTIONS) * 2,
            actiondim=len(INTERVENTIONS),
            device=self.device
        )

        # 2. initialize environments and logging
        self.validation_envs = self._create_validation_envs()
        self.csv_logger = CSVLogger(log_dir=args.logdir)
    
    def _initialize_student(self):
        # logic to create the initial PPO student model
        pass

    def pretrain_teacher(self):
        """
        cleaner pretraining loop. student model is passed but remains frozen.
        NOTE: is that the right approach? shouldn't the student model respond to the interventions provided by the teacher?
        it is this adaptive loop that will help the teacher learn.
        """
        logging.info("Starting teacher pretraining...")
        # all pretraining logic is self-contained here.
        # computes states, selects actions, calculates rewards.
        # updates the teacher network against a static student.
        pass

    def train(self):
        """
        main meta-learning loop.
        """
        logging.info("Starting AutoCaLC meta-training...")

        last_validation_perf = self.run_validation_protocol(self.student)

        for meta_step in range(self.args.meta_episodes):
            # the entire meta-episode logic is now a clean method call
            last_validation_perf = self.run_meta_episode(meta_step, last_validation_perf)
    
    def run_meta_episode(self, meta_step, last_validation_perf):
        """
        executes a single, complete teacher-student interaction.
        """
        # 1. get current state
        meta_state = self.get_teacher_state(self.student)

        # 2. teacher selects intervention
        epsilon = get_exploration_rate(meta_step, self.args.meta_episodes)
        action_idx = self.teacher.select_action(meta_state, epsilon)

        # 3. student trains on the intervention
        self.train_student_on_intervention(self.student, action_idx, meta_step)

        # 4. evaluate student and get meta-reward
        current_validation_perf = self.run_validation_protocol(self.student)
        meta_reward = current_validation_perf - last_validation_perf    # learning progress

        # 5. get next state and update teacher
        next_meta_state = self.get_teacher_state(self.student)
        self.teacher.replay_buffer.push(meta_state, action_idx, meta_reward, next_meta_state)
        self.teacher.train_step()
        
        return current_validation_perf
    
    # include other helper functions like get_teacher_state, train_student, etc.


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    autocalc_framework = AutoCaLC(args)

    if args.teacher_pretrain_steps > 0:
        autocalc_framework.pretrain_teacher()
        if args.save_pretrained_teacher():
            autocalc_framework.teacher.save(args.save_pretrained_teacher)
    
    if not args.teacher_pretrain_only:
        if args.load_pretrained_teacher:
            autocalc_framework.teacher.load(args.load_pretrained_teacher)
        autocalc_framework.train()


# ----------------------------------------------------------------------------------------------

# obtaining the exploration rate
def get_exploration_rate(meta_step, meta_episodes):
    """
    advanced exploration strategy with three phases:
    1. initial pure exploration phase
    2. guided exploration phase with faster decay
    3. fine-tuning phase with minimal exploration
    """
    # phase 1: initial pure exploration (first 10% of episodes)
    if meta_step < meta_episodes * 0.1:
        return 1.0

    # phase 2: guided exploration (next 60% of episodes)
    elif meta_step < meta_episodes * 0.7:
        # start at 0.8 and decay to 0.2 over this phase
        phase_progress = (meta_step - meta_episodes * 0.1) / (meta_episodes * 0.6)
        return 0.8 - 0.6 * phase_progress
    
    # phase 3: fine-tuning (last 30% of episodes)
    else:
        # low exploration rate for fine-tuning
        return 0.2 - 0.15 * ((meta_step - meta_episodes * 0.7) / (meta_episodes * 0.3))

# setting random seed
def set_seed(seed):
    """set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # ensuring deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"random seed set to {seed}")

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
        logging.info(f"     Processing intervention {i+1}/{len(interventions)}: {intervention['type']}")

        # 1. Test the reward performance
        reward_metrics = test_intervention_performance(
            student_model,
            intervention,
            task_name,
            num_episodes=5,
            seed=seed + i
        )
        rewards.append(reward_metrics['avg_reward'])
        logging.info(f"     {intervention['type']} reward: {reward_metrics['avg_reward']:.4f}")

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
        logging.info(f"     {intervention['type']} CM score: {cm_score:.4f}")
    
    # normalize reward scores
    reward_array = np.array(rewards, dtype=np.float32)
    reward_mean = np.mean(reward_array)
    reward_std = np.std(reward_array)
    if reward_std > 1e-8:
        normalized_rewards = (reward_array - reward_mean) / reward_std
    else:
        normalized_rewards = reward_array - reward_mean
    
    # normalize CM scores
    cm_array = np.array(cm_scores, dtype=np.float32)  # Fixed: was using rewards instead of cm_scores
    cm_mean = np.mean(cm_array)
    cm_std = np.std(cm_array)
    if cm_std > 1e-8:
        normalized_cm = (cm_array - cm_mean) / cm_std
    else:
        normalized_cm = cm_array - cm_mean
    
    # interleave rewards and CM scores
    interleaved_state = np.zeros(len(interventions) * 2, dtype=np.float32)
    interleaved_state[0::2] = normalized_rewards    # even indices for rewards
    interleaved_state[1::2] = normalized_cm         # odd indices for CM scores
    
    logging.info(f"Raw rewards: {reward_array}")
    logging.info(f"Raw CM scores: {cm_array}")
    logging.info(f"Normalized rewards: {normalized_rewards}")
    logging.info(f"Normalized CM scores: {normalized_cm}")
    logging.info(f"Interleaved state shape: {interleaved_state.shape}")
    
    return interleaved_state

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

def test_all_interventions_for_teacher(student_model, task_name, stage_num, args, csv_logger, cumulative_timesteps, selected_intervention_type):
    """Test student model on all interventions to gather data for teacher state and logging"""
    logging.info(f"[AUTOCALC] Testing all interventions at meta-episode {stage_num}")
    
    # Test each intervention type including none
    for intervention in INTERVENTIONS + [None]:
        int_type = intervention['type'] if intervention is not None else 'none'
        
        # Test performance on this intervention
        metrics = test_intervention_performance(
            student_model=student_model,
            intervention=intervention,
            task_name=task_name,
            num_episodes=getattr(args, 'test_episodes', 10),
            seed=args.seed + stage_num * 100 + (INTERVENTIONS.index(intervention) if intervention in INTERVENTIONS else 0)
        )
        
        # Calculate CM score if needed for teacher state
        env = create_environment(task_name, intervention, seed=args.seed + stage_num * 100)
        cm_score = evaluate_cm_score(
            env, 
            student_model, 
            max_episodes=5, 
            device='cuda' if torch.cuda.is_available() else 'cpu',
            intervention_type=int_type, 
            seed=args.seed + stage_num * 100
        )
        env.close()
        
        # Log results to CSV
        if csv_logger:
            csv_logger.log_intervention_test(
                stage=stage_num,
                intervention_type=int_type,
                test_avg_reward=metrics['avg_reward'],
                test_success_rate=metrics['success_rate'],
                test_avg_length=metrics['avg_length'],
                selected=(int_type == selected_intervention_type),  # Mark which intervention was selected
                cumulative_timesteps=cumulative_timesteps,
                cm_score=cm_score,
                baseline_type="autocalc"
            )

# the teacher dqn agent
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

# TODO: modify this to leakyrelu if you get a vanishing gradient issue
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
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, device='cpu', buffer_size=50000, batch_size=64, target_update=5):
        self.device = device
        self.q_net = DQN(state_dim, action_dim).to(device)
        self.target_net = deepcopy(self.q_net)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.replay_buffer = ReplayBuffer(buffer_size, state_dim, device)
        self.update_count = 0
    
    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.q_net.net[-1].out_features - 1)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return int(q_values.argmax().item())
    
    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        s, a, r, s_ = self.replay_buffer.sample(self.batch_size)
        q_values = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            max_next_q = self.target_net(s_).max(1)[0]
            target = r + self.gamma * max_next_q
        
        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        # self.update_count += 1
        # if self.update_count % self.target_update == 0:
        #     self.target_net.load_state_dict(self.q_net.state_dict())
        
        # replacing hard update with:
        tau = 0.005
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return loss.item()

    def save(self, path):
        """save the teacher model and optimizer state"""
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'buffer': self.replay_buffer.buffer,
            'update_count': self.update_count
        }, path)
        logging.info(f"teacher model saved to {path}")
    
    def load(self, path):
        """load the teacher model and optimizer state"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No teacher model found at {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # restore the replay buffer if it exists in the checkpoint
        if 'buffer' in checkpoint:
            self.replay_buffer.buffer = checkpoint['buffer']
        if 'update_count' in checkpoint:
            self.update_count = checkpoint['update_count']
        
        logging.info(f"Teacher model loaded from {path}")

# our pretraining logic
def pretrain_teacher(teacher, student_model, args, device='cpu'):
    """
    Pretrain the teacher agent using a frozen student model

    Args:
        teacher: TeacherDQNAgent instance
        student_model: Frozen PPO student model
        args: command line arguments
        device: the device to use (cpu or cuda)
    """

    logging.info(f"=== Starting teacher pretraining for {args.teacher_pretrain_steps} steps ===")

    # initialize metrics tracking
    pretraining_losses = []
    pretraining_rewards = []
    pretraining_interventions = []

    # set up csv logger for pretraining data
    pretrain_log_path = os.path.join(args.log_dir, "teacher_pretraining.csv")
    with open(pretrain_log_path, 'w') as f:
        f.write("step,intervention,meta_reward,loss\n")
    
    # number of teacher training steps
    num_pretrain_steps = args.teacher_pretrain_steps

    # use fixed validation env for consistent meta-reward evaluation
    pretrain_validation_dir = os.path.join(args.log_dir, "pretrain_validation")
    os.makedirs(pretrain_validation_dir, exist_ok=True)
    validation_csv_logger = CSVLogger(pretrain_validation_dir)

    # pre-training loop
    for step in range(1, num_pretrain_steps + 1):
        # 1. compute current meta-state (with fixed student)
        current_meta_state = get_teacher_state(
            student_model,
            args.task,
            INTERVENTIONS,
            device=device,
            seed=args.seed + step
        )

        # 2. teacher selects an intervention
        # we would use an exploratory epsilon during pretraining
        epsilon = max(0.1, 1.0 - (step / num_pretrain_steps) * 0.9)
        selected_intervention_idx = teacher.select_action(current_meta_state, epsilon)
        selected_intervention = INTERVENTIONS[selected_intervention_idx]
        selected_intervention_type = selected_intervention['type']

        # 3. no student training happens here! we simply evaluate the student on this intervention

        # 4. run validation to measure generalization
        validation_metrics = run_validation_protocol(
            student_model,
            args.task,
            step,
            args,
            validation_csv_logger,
            0   # no cumulative timesteps since the student is not training
        )

        # 5. calculate meta-reward (same formula as the main loop)
        # NOTE: i removed the weighting for success rate, keeping it simple by only considering val reward
        meta_reward = (0 * validation_metrics['validation_success_rate']) + validation_metrics['validation_avg_reward']

        # 6. compute next meta-state (with fixed student)
        next_meta_state = get_teacher_state(
            student_model,
            args.task,
            INTERVENTIONS,
            device=device,
            seed=args.seed + step + 1
        )

        # 7. store experience and update the teacher
        teacher.replay_buffer.push(current_meta_state, selected_intervention_idx, meta_reward, next_meta_state)
        loss = teacher.train_step()

        # 8. track metrics
        pretraining_interventions.append(selected_intervention_type)
        pretraining_rewards.append(meta_reward)
        pretraining_losses.append(loss if loss is not None else 0)

        # 9. log progress
        if step % 10 == 0 or step == 1 or step == num_pretrain_steps:
            avg_loss = np.mean([l for l in pretraining_losses[-10:] if l is not None])
            avg_reward = np.mean(pretraining_rewards[-10:])
            logging.info(f"Pretraining step {step}/{num_pretrain_steps}: " +
                         f"epsilon={epsilon:.2f}, " +
                         f"intervention={selected_intervention_type}, " +
                         f"meta-reward={meta_reward:.2f}, " +
                         f"avg_loss={avg_loss:.4f}")
        
        # 10. log to csv
        with open(pretrain_log_path, 'a') as f:
            f.write(f"{step},{selected_intervention_type},{meta_reward:.4f},{loss if loss is not None else 'None'}\n")
        
        # 11. log to wandb if enabled
        if args.use_wandb:
            wandb.log({
                'pretrain_step': step,
                'pretrain_teacher_action': selected_intervention_type,
                'pretrain_meta_reward': meta_reward,
                'pretrain_teacher_loss': loss,
                'pretrain_epsilon': epsilon
            })
        
        # 12. save intermediate models
        if step % 1000 == 0 or step == num_pretrain_steps:
            intermediate_path = os.path.join(args.log_dir, f"teacher_pretrained_{step}_steps.pt")
            teacher.save(intermediate_path)

    # save final pretrained teacher model
    if args.save_pretrained_teacher:
        teacher.save(args.save_pretrained_teacher)
    else:
        default_save_path = os.path.join(args.log_dir, f"teacher_pretrained_{num_pretrain_steps}_steps.pt")
        teacher.save(default_save_path)
    
    logging.info(f"=== Teacher pretraining completed for {num_pretrain_steps} steps ===")

    # return statistics about pretraining
    return {
        'avg_reward': np.mean(pretraining_rewards),
        'avg_loss': np.mean([l for l in pretraining_losses if l is not None]),
        'intervention_counts': {
            int_type: pretraining_interventions.count(int_type)
            for int_type in set(pretraining_interventions)
        }
    }

# the eval function
def evaluate_autocalc_performance(log_dir, task_name, eval_model, seed=0, max_episode_length=250, skip_frame=3, num_episodes=10):
    """made similar to baselines evaluation"""
    set_seed(seed)
    logging.info("Running AutoCaLC evaluation...")

    # create env
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

    # load the final model
    model_path = os.path.join(log_dir, eval_model)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}")
    
    model = PPO.load(model_path)

    # basic episode evaluation
    logging.info("\nFirst phase of evaluation:")
    all_rewards, all_successes = [], []
    for ep in range(num_episodes):
        obs = env.reset()
        if hasattr(env, 'seed'):
            env.seed(seed + ep)
        done = False
        total_reward, successes = 0.0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if isinstance(info, dict) and 'success' in info:
                successes += int(info['success'])
        logging.info(f"Episode {ep + 1}: reward = {total_reward:.2f}, success = {successes}")
        all_rewards.append(total_reward)
        all_successes.append(successes)
    
    logging.info(f"\nMean reward: {np.mean(all_rewards):.2f}")
    logging.info(f"Mean success rate: {np.mean(all_successes):.2f}")

    # benchmark eval (same as baselines)
    logging.info("\nGenerating benchmark evaluation and viz...")
    if task_name not in TASK_BENCHMARKS:
        logging.error(f"No benchmark available for task: '{task_name}'. Supported: {SUPPORTED_TASKS}")
        return
    benchmark = TASK_BENCHMARKS[task_name]

    # use the causalworld benchmark evaluation
    evaluation = EvaluationPipeline(
        evaluation_protocols=benchmark['evaluation_protocols'],
        task_params={'task_generator_id': task_name},
        world_params={'skip_frame': 3, 'action_mode': 'joint_torques'},
        visualize_evaluation=False
    )

    def policy_fn(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action
    
    scores_model = evaluation.evaluate_policy(policy_fn, fraction=0.005)

    logging.info("\nEvaluation results:")
    logging.info(scores_model)

    # save the benchmark results
    import json
    benchmark_path = os.path.join(log_dir, "benchmark_results.json")
    with open(benchmark_path, 'w') as f:
        json.dump({
            'final_evals': scores_model
        }, f, indent=2)
    logging.info(f"Final evals saved to: {benchmark_path}")

    # generate the visualizations
    plots_dir = os.path.join(log_dir, "plots")
    vis.generate_visual_analysis(plots_dir, experiments={task_name: scores_model})
    logging.info(f"Visualization saved to: {plots_dir}")

# ========================
# main meta-learning loop
# ========================
def main():
    # Get arguments from the global parser (already parsed in __main__)
    import sys
    # Re-parse arguments in main for clarity
    parser = argparse.ArgumentParser(description="Meta-learning teacher-student framework for CausalWorld")
    parser.add_argument('--task', type=str, default='pushing', help='CausalWorld task to train on')
    parser.add_argument('--student_train_steps', type=int, default=50000, help='Timesteps for each student training block')
    parser.add_argument('--meta_episodes', type=int, default=50, help='Number of meta-episodes')
    parser.add_argument('--student_pretrained_path', type=str, default=None, help='Path to pretrained PPO model (optional)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--use_wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--log_dir', type=str, default='logs/autocalc', help='Log directory')
    parser.add_argument('--test_episodes', type=int, default=10, help='Episodes for testing each intervention')
    parser.add_argument('--validation_episodes', type=int, default=10, help='Episodes per validation environment')
    parser.add_argument('--skip_frame', type=int, default=3, help='Frame skip for environment')
    parser.add_argument('--eval', action='store_true', help='Evaluate the model')
    parser.add_argument('--teacher_seed', type=int, default=0, help='Seed for the teacher agent')
    parser.add_argument('--student_seed', type=int, default=0, help='Seed for the student agent initial policy')
    parser.add_argument('--teacher_pretrain_steps', type=int, default=0, help='Number of steps to pre-train the teacher')
    parser.add_argument('--save_pretrained_teacher', type=str, default=None, help='Path to save the pretrained teacher model')
    parser.add_argument('--load_pretrained_teacher', type=str, default=None, help='Path to load a pretrained teacher model')
    parser.add_argument('--teacher_pretrain_only', action='store_true', help='Only pretrain the teacher, then exit (no student training)')
    parser.add_argument('--device_id', type=int, default=6, choices=[6, 7], help='GPU device ID to use (6 or 7)')

    args = parser.parse_args()

    # Set up GPU device
    device = torch.device(f"cuda:{args.device_id}")
    torch.cuda.set_device(args.device_id)
    logging.info(f"Using GPU device: {device}")
    
    # Store device in args for access throughout the script
    args.device = device

    # the initial setup phase
    os.makedirs(args.log_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(os.path.join(args.log_dir, 'autocalc.log')), logging.StreamHandler()])

    # Initialize csv logger
    csv_logger = CSVLogger(args.log_dir)

    if args.student_pretrained_path is None:
        args.student_pretrained_path = f'models/ppo_{args.task}_sb3/final_model.zip'
        logging.info(f"Auto-determined pretrained path: {args.student_pretrained_path}")
    
    if args.use_wandb:
        wandb.init(
            project=f'autocalc-{args.task}',
            name=f'autocalc_{args.task}_seed{args.seed}',
            config=vars(args),
            tags=[args.task, 'autocalc', 'meta-learning']
        )
    
    logging.info(f"Starting AutoCaLC training for task '{args.task}' on device '{args.device}'")

    # --- 1. Initialization ---
    logging.info("=== Initializing student and teacher agents ===")

    # loading the base pretrained model
    student_model = PPO.load(args.student_pretrained_path, device=args.device)
    logging.info(f"Loaded base pretrained model from {args.student_pretrained_path}")

    # setting the student's random seed for reproducible weight initialization
    set_seed(args.student_seed)                         # set the global seeds first
    student_model.set_random_seed(args.student_seed)    # set the model-specific seed
    
    # additional seed setting for the policy and value networks
    if hasattr(student_model.policy, 'action_net'):
        torch.manual_seed(args.student_seed)
        for param in student_model.policy.parameters():
            if param.requires_grad:
                # re-initialize parameters with the fixed seed
                if len(param.shape) > 1:
                    torch.nn.init.xavier_uniform_(param)
                else:
                    torch.nn.init.zeros_(param)
    
    logging.info(f"Loaded student PPO from {args.student_pretrained_path} with seed {args.student_seed} for reproducible weights")

    # initialize teacher with its own seed
    set_seed(args.teacher_seed)
    teacher = TeacherDQNAgent(
        state_dim=len(INTERVENTIONS) * 2,
        action_dim=len(INTERVENTIONS),
        device=args.device
    )
    logging.info(f"Teacher DQN initialized with seed {args.teacher_seed}")

    # load the pretrained teacher if specified
    if args.load_pretrained_teacher:
        teacher.load(args.load_pretrained_teacher)
        logging.info(f"Loaded pretrained teaher from {args.load_pretrained_teacher}")
    
    # run teacher pretraining if requested
    if args.teacher_pretrain_steps > 0:
        pretrain_stats = pretrain_teacher(teacher, student_model, args, args.device)
        logging.info(f"Teacher pretraining completed with stats: {pretrain_stats}")

        # if only pretraining was requested, then you would exit here
        if args.teacher_pretrain_only:
            logging.info("Teacher pretraining only mode - exiting without student training")
            if args.use_wandb:
                wandb.finish()
            return

    # reset to student seed for subsequent operations
    set_seed(args.student_seed)

    # track cumulative timesteps
    cumulative_timesteps = 0
    
    # --- Zero-shot validation (meta-episode 0) ---
    logging.info("--- Running zero-shot validation (meta-episode 0) ---")
    initial_validation_metrics = run_validation_protocol(
        student_model, args.task, 0, args, csv_logger, cumulative_timesteps
    )
    last_validation_metrics = initial_validation_metrics
    logging.info(f"Zero-shot validation complete: success rate = {initial_validation_metrics['validation_success_rate']:.3f}, avg reward = {initial_validation_metrics['validation_avg_reward']:.3f}")

    # track the best model based on validation reward
    best_validation_reward = initial_validation_metrics['validation_avg_reward']
    best_model_meta_step = 0
    best_model_path = os.path.join(args.log_dir, "best_student_model.zip")

    # save initial model as the best so far
    student_model.save(best_model_path)
    logging.info(f"Saved initial model as best (reward: {best_validation_reward}:.4f)")
    
    # --- 2. Main meta-learning loop ---
    for meta_step in range(args.meta_episodes):
        logging.info(f"\n{'='*20} Meta-Episode {meta_step + 1}/{args.meta_episodes} {'='*20}")

        # 3. Compute current meta-state
        current_meta_state = get_teacher_state(student_model, args.task, INTERVENTIONS, device=device, seed=args.seed + meta_step)

        # 4. Teacher selects an intervention
        # NOTE: here I am skipping teacher exploration since it is already pretrained
        # epsilon = get_exploration_rate(meta_step, args.meta_episodes)
        epsilon = 0.1
        selected_intervention_idx = teacher.select_action(current_meta_state, epsilon)
        selected_intervention = INTERVENTIONS[selected_intervention_idx]
        selected_intervention_type = selected_intervention['type']
        logging.info(f"Teacher chose intervention: '{selected_intervention_type}' (Epsilon: {epsilon:.2f})")

        # 5. Student trains on that intervention
        train_env = create_environment(args.task, selected_intervention, seed=args.seed + meta_step * 100)
        vec_env = DummyVecEnv([lambda: train_env])
        vec_env = VecMonitor(vec_env, filename=os.path.join(args.log_dir, f'autocalc_monitor_stage{meta_step+1}.csv'))
        
        # Set up training monitoring
        reward_monitor = RewardMonitorCallback(
            selected_intervention_type, csv_logger, meta_step+1, cumulative_timesteps, baseline_type="autocalc"
        )
        
        # Create callback list
        callback_list = CallbackList([
            reward_monitor,
            WandbCallback(
                gradient_save_freq=100,
                model_save_path=args.log_dir if args.use_wandb else None,
                verbose=2
            ) if args.use_wandb else None
        ])
        callback_list.callbacks = [cb for cb in callback_list.callbacks if cb is not None]
        
        # Train the student
        student_model.set_env(vec_env)
        student_model.learn(
            total_timesteps=args.student_train_steps,
            callback=callback_list,
            reset_num_timesteps=False
        )
        
        # Update cumulative timesteps
        cumulative_timesteps += args.student_train_steps
        
        # 6. Test on all interventions for logging
        test_all_interventions_for_teacher(
            student_model, args.task, meta_step+1, args, csv_logger, cumulative_timesteps, selected_intervention_type
        )
        
        # 7. Evaluate new generalization performance
        current_validation_metrics = run_validation_protocol(
            student_model, args.task, meta_step+1, args, csv_logger, cumulative_timesteps
        )

        # ... checking if this is the best model so far
        current_validation_reward = current_validation_metrics['validation_avg_reward']
        if current_validation_reward > best_validation_reward:
            best_validation_reward = current_validation_reward
            best_model_meta_step = meta_step + 1

            # save as best model
            student_model.save(best_model_path)
            logging.info(f"New best model at meta-step {meta_step+1} with validation reward: {best_validation_reward:.4f}")

            # also log to wandb if enabled
            if args.use_wandb:
                wandb.run.summary["best_validation_reward"] = best_validation_reward
                wandb.run.summary["best_model_meta_step"] = best_model_meta_step

        # 8. Calculate the meta-reward using the new formula
        # NOTE: i removed the weighting from success rate, keeping the meta-reward formulation simple
        meta_reward = (0 * current_validation_metrics['validation_success_rate']) + current_validation_metrics['validation_avg_reward']
        logging.info(f"Meta-Reward: {meta_reward:.4f} (Success Rate: {current_validation_metrics['validation_success_rate']:.3f}, Avg Reward: {current_validation_metrics['validation_avg_reward']:.3f})")

        # 9. Compute the next meta-state
        next_meta_state = get_teacher_state(student_model, args.task, INTERVENTIONS, device=args.device, seed=args.seed + meta_step + 1)

        # 10. Store experience and update the teacher
        teacher.replay_buffer.push(current_meta_state, selected_intervention_idx, meta_reward, next_meta_state)
        teacher_loss = teacher.train_step()
        logging.info(f"Teacher training step complete. Loss: {teacher_loss if teacher_loss is not None else 'N/A'}")

        # 11. Log to wandb
        if args.use_wandb:
            wandb.log({
                'meta_episode': meta_step + 1,
                'teacher_action': selected_intervention_type,
                'teacher_reward': meta_reward,
                'student_validation_success_rate': current_validation_metrics['validation_success_rate'],
                'student_validation_reward': current_validation_metrics['validation_avg_reward'],
                'teacher_epsilon': epsilon,
                'teacher_loss': teacher_loss,
                'cumulative_timesteps': cumulative_timesteps,
                **{f"cm_score_{inter['type']}": score for inter, score in zip(INTERVENTIONS, current_meta_state[1::2])}
            })
        
        stage_model_path = os.path.join(args.log_dir, f"student_model_stage_{meta_step+1}.zip")
        student_model.save(stage_model_path)
        logging.info(f"Saved intermediate model for stage {meta_step+1} to {stage_model_path}")
        
        # Clean up
        train_env.close()
    
    # --- And now the final step ---
    final_model_path = os.path.join(args.log_dir, "final_student_model.zip")
    student_model.save(final_model_path)
    logging.info(f"Meta-RL training complete! Final student model saved to {final_model_path}")

    # log final best model info
    logging.info(f"Best student model was from meta-step {best_model_meta_step} with validation reward: {best_validation_reward:.4f}")
    logging.info(f"Best model saved to: {best_model_path}")

    if args.use_wandb:
        wandb.run.summary["best_validation_reward"] = best_validation_reward
        wandb.run.summary["best_model_meta_step"] = best_model_meta_step
        wandb.run.summary["best_model_path"] = best_model_path
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Meta-RL Teacher-Student Curriculum (AutoCaLC)")
    parser.add_argument('--eval', action='store_true', help='Evaluate the model')
    parser.add_argument('--task', type=str, default='pushing', help='CausalWorld task name')
    parser.add_argument('--student_train_steps', type=int, default=50000, help='Timesteps per student training block')
    parser.add_argument('--meta_episodes', type=int, default=50, help='Number of meta-episodes (teacher steps)')
    parser.add_argument('--student_pretrained_path', type=str, default=None, help='Path to pretrained PPO model (optional)')
    parser.add_argument('--pretrained_eval', type=str, help='Path to eval model')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--use_wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--log_dir', type=str, default=None, help='Log directory (will be auto-generated if not specified)')
    parser.add_argument('--test_episodes', type=int, default=10, help='Episodes for testing each intervention')
    parser.add_argument('--validation_episodes', type=int, default=10, help='Episodes per validation environment')
    parser.add_argument('--skip_frame', type=int, default=3, help='Frame skip for environment')
    parser.add_argument('--teacher_seed', type=int, default=0, help='Seed for the teacher agent')
    parser.add_argument('--student_seed', type=int, default=0, help='Seed for the student agent initial policy')
    parser.add_argument('--teacher_pretrain_steps', type=int, default=0, help='Number of steps to pre-train the teacher')
    parser.add_argument('--save_pretrained_teacher', type=str, default=None, help='Path to save the pretrained teacher model')
    parser.add_argument('--load_pretrained_teacher', type=str, default=None, help='Path to load a pretrained teacher model')
    parser.add_argument('--teacher_pretrain_only', action='store_true', help='Only pretrain the teacher, then exit (no student training)')
    parser.add_argument('--device_id', type=int, default=6, choices=[6, 7], help='GPU device ID to use (6 or 7)')
    args = parser.parse_args()

    # Set default log directory if not specified
    if args.log_dir is None:
        if args.eval:
            raise ValueError("--log_dir must be specified for evaluation")
        else:
            args.log_dir = f'logs/autocalc_{args.task}_seed{args.seed}'

    if args.eval:
        evaluate_autocalc_performance(args.log_dir, task_name=args.task, eval_model=args.pretrained_eval, seed=args.seed)
    else:
        main()
