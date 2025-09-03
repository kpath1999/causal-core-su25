import os
import csv
import time
import numpy as np
import pandas as pd
from stable_baselines3.common.logger import Logger, configure
from stable_baselines3.common.monitor import Monitor
import logging

class ConsolidatedLogger:
    """
    A centralized logging system that maintains single CSV files across all training stages
    and standardizes metrics collection across different baseline methods.
    """
    def __init__(self, log_dir, algorithm_name, task_name):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.algorithm_name = algorithm_name
        self.task_name = task_name
        
        # Define consolidated log files
        self.progress_csv_path = os.path.join(log_dir, 'consolidated_progress.csv')
        self.monitor_csv_path = os.path.join(log_dir, 'consolidated_monitor.csv')
        self.intervention_log_path = os.path.join(log_dir, 'intervention_log.csv')
        self.validation_log_path = os.path.join(log_dir, 'validation_metrics.csv')
        
        # Initialize CSV files with headers if they don't exist
        self._initialize_csv_files()
        
        # Track current stage
        self.current_stage = 0
        self.cumulative_timesteps = 0
        
        # Configure SB3 logger to use our consolidated directory
        self.sb3_logger = None
    
    def _initialize_csv_files(self):
        """Initialize CSV files with headers if they don't exist"""
        # Progress CSV (for training metrics)
        if not os.path.exists(self.progress_csv_path):
            with open(self.progress_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'stage', 'total_timesteps', 'time_elapsed', 'fps', 
                    'explained_variance', 'entropy', 'learning_rate',
                    'n_updates', 'policy_loss', 'value_loss', 'approx_kl',
                    'clip_fraction', 'clip_range', 'reward_mean', 'reward_std',
                    'reward_min', 'reward_max', 'ep_length_mean', 'success_rate'
                ])
        
        # Monitor CSV (for episode rewards)
        if not os.path.exists(self.monitor_csv_path):
            with open(self.monitor_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'stage', 'r', 'l', 't', 'success'
                ])
        
        # Intervention log
        if not os.path.exists(self.intervention_log_path):
            with open(self.intervention_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'stage', 'intervention_type', 'test_avg_reward',
                    'test_success_rate', 'test_avg_length', 'selected',
                    'cumulative_timesteps', 'cm_score'
                ])
        
        # Validation metrics log
        if not os.path.exists(self.validation_log_path):
            with open(self.validation_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'stage', 'cumulative_timesteps', 'avg_reward', 'success_rate',
                    'avg_episode_length', 'timestamp'
                ])
    
    def start_new_stage(self, stage_num, intervention_type=None):
        """Record the start of a new training stage"""
        self.current_stage = stage_num
        
        if intervention_type:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(self.intervention_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, stage_num, intervention_type, "", "", "", "True",
                    self.cumulative_timesteps, ""
                ])
        
        # Configure SB3 logger for this stage (will write to our consolidated files)
        tmp_log_dir = os.path.join(self.log_dir, f"tmp_stage_{stage_num}")
        os.makedirs(tmp_log_dir, exist_ok=True)
        self.sb3_logger = configure(tmp_log_dir, ["stdout", "csv"])
        
        return self.sb3_logger
    
    def update_cumulative_timesteps(self, additional_timesteps):
        """Update the total timesteps counter across all stages"""
        self.cumulative_timesteps += additional_timesteps
    
    def merge_progress_csv(self, stage_csv_path):
        """Merge a stage-specific progress.csv into the consolidated file"""
        if not os.path.exists(stage_csv_path):
            return
        
        try:
            stage_df = pd.read_csv(stage_csv_path)
            # Add stage information
            stage_df['stage'] = self.current_stage
            stage_df['total_timesteps'] += self.cumulative_timesteps - len(stage_df)
            
            # Append to consolidated file
            stage_df.to_csv(self.progress_csv_path, mode='a', header=False, index=False)
        except Exception as e:
            logging.error(f"Error merging progress CSV: {e}")
    
    def merge_monitor_csv(self, stage_monitor_path):
        """Merge a stage-specific monitor.csv into the consolidated file"""
        if not os.path.exists(stage_monitor_path):
            return
        
        try:
            # Skip the first two rows (header and version info)
            stage_df = pd.read_csv(stage_monitor_path, skiprows=1)
            # Add stage information
            stage_df['stage'] = self.current_stage
            
            # Append to consolidated file
            stage_df.to_csv(self.monitor_csv_path, mode='a', header=False, index=False)
        except Exception as e:
            logging.error(f"Error merging monitor CSV: {e}")
    
    def log_validation_metrics(self, metrics):
        """Log validation metrics to the consolidated validation file"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.validation_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.current_stage,
                self.cumulative_timesteps,
                metrics.get('avg_reward', 0),
                metrics.get('success_rate', 0),
                metrics.get('avg_episode_length', 0),
                timestamp
            ])
    
    def log_intervention_selection(self, intervention_type, test_metrics, cm_score=None, selected=True):
        """Log an intervention selection with test metrics"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.intervention_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, 
                self.current_stage,
                intervention_type,
                test_metrics.get('avg_reward', ''),
                test_metrics.get('success_rate', ''),
                test_metrics.get('avg_episode_length', ''),
                selected,
                self.cumulative_timesteps,
                cm_score or ''
            ])

class MonitorWithSuccess(Monitor):
    """Extended Monitor wrapper that also tracks success flag"""
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        
        if done:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)
            ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
            
            # Add success info if available
            if isinstance(info, dict) and 'success' in info:
                ep_info["success"] = float(info['success'])
            else:
                ep_info["success"] = 0.0
                
            self.episode_returns.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            
            ep_info.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(ep_info)
                
            info["episode"] = ep_info
        
        return observation, reward, done, info