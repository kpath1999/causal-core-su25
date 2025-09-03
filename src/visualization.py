"""
Enhanced visualization module for curriculum learning analysis.
Based on the provided Colab notebook with improvements for automation and flexibility.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import logging

# Set up modern aesthetics
try:
    plt.style.use('seaborn-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default')
        print("Using default matplotlib style")

sns.set_palette("husl")

# Enhanced font settings
font_config = {'size': 14, 'family': 'sans-serif'}
plt.rc('font', **font_config)
plt.rc('axes', labelsize=12, titlesize=16)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('legend', fontsize=12)
plt.rc('figure', titlesize=18)

class EnhancedTrainingVisualizer:
    """Enhanced visualization class with modern aesthetics and comprehensive analysis."""
    
    def __init__(self, log_dir: str, output_dir: Optional[str] = None):
        """
        Initialize the enhanced visualizer.
        
        Args:
            log_dir: Directory containing training logs
            output_dir: Directory to save visualizations (defaults to log_dir/plots)
        """
        self.log_dir = log_dir
        self.output_dir = output_dir or os.path.join(log_dir, 'enhanced_plots')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Modern color palette for different heuristics
        self.colors = {
            'greedy': '#2E86AB',      # Professional blue
            'cm': '#A23B72',          # Deep magenta  
            'causal_mismatch': '#A23B72',  # Alternative name
            'none': '#F18F01',        # Vibrant orange
            'random': '#C73E1D',      # Bold red
            'rnd': '#6A994E',         # Forest green
            'count': '#7209B7'        # Purple
        }
        
        # Define comprehensive metrics to plot
        self.training_metrics = [
            ('Mean Episode Reward', 'mean_episode_reward', 'blue'),
            ('Mean Episode Length', 'mean_episode_length', 'orange'),
            ('Explained Variance', 'explained_variance', 'purple'),
            ('Value Loss', 'value_loss', 'brown'),
            ('Policy Std', 'policy_std', 'cyan'),
            ('Entropy Loss', 'entropy_loss', 'magenta'),
            ('Clip Fraction', 'clip_fraction', 'green'),
            ('Approx KL Divergence', 'approx_kl_divergence', 'red')
        ]
        
        # Validation metrics
        self.validation_metrics = [
            'validation_avg_reward',
            'validation_success_rate', 
            'validation_avg_length'
        ]
        
        # Evaluation metrics for radar plots
        self.radar_metrics = [
            'mean_full_integrated_fractional_success',
            'mean_last_integrated_fractional_success', 
            'mean_last_fractional_success'
        ]

    def add_cumulative_iteration(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cumulative iteration column to handle stage resets."""
        df = df.sort_values(by=['stage', 'iteration'])
        
        cumulative_offset = 0
        cumulative_iterations = []
        previous_iteration = None
        
        for idx, iteration in enumerate(df['iteration']):
            if previous_iteration is not None and iteration < previous_iteration:
                # When iteration resets (current iteration < previous iteration)
                cumulative_offset += previous_iteration
            cumulative_iterations.append(iteration + cumulative_offset)
            previous_iteration = iteration
        
        df['cumulative_iteration'] = cumulative_iterations
        return df

    def add_intervention_markers(self, ax, x, y_data, df, intervention_col='intervention_type', 
                               stage_col='stage', alpha=0.5):
        """Add vertical lines and labels for intervention changes."""
        if intervention_col in df.columns and stage_col in df.columns:
            prev_stage = None
            for idx, row in df.iterrows():
                if idx == 0 or row[stage_col] != prev_stage:
                    ax.axvline(x=row['cumulative_iteration'], color='green', 
                             linestyle='--', alpha=alpha, linewidth=1.5)
                    # Position label at 90% of the max y-value to avoid overlap
                    ymax = np.max(y_data) if len(y_data) > 0 else 1
                    ax.text(row['cumulative_iteration'], ymax * 0.9, 
                           f"{row[intervention_col]}", rotation=90, fontsize=8, 
                           ha='right', va='top', alpha=0.8)
                    prev_stage = row[stage_col]

    def plot_single_heuristic_analysis(self, progress_path: str, heuristic_name: str, 
                                     validation_path: Optional[str] = None):
        """Create comprehensive analysis plots for a single heuristic."""
        try:
            # Load and process data
            df = pd.read_csv(progress_path)
            df = self.add_cumulative_iteration(df)
            
            # Load validation data if available
            df_val = None
            if validation_path and os.path.exists(validation_path):
                df_val = pd.read_csv(validation_path)
                logging.info(f"Loaded validation data from {validation_path}")
            
            # Create 4x2 subplot grid
            fig, axes = plt.subplots(4, 2, figsize=(16, 20))
            fig.suptitle(f'{heuristic_name.title()} Heuristic: Training Analysis with Interventions', 
                        fontsize=20, y=0.98)
            
            x = df['cumulative_iteration']
            intervention_col = 'intervention_type'
            stage_col = 'stage'
            
            # Plot each metric
            for i, (title, metric_col, color) in enumerate(self.training_metrics):
                row = i // 2
                col = i % 2
                ax = axes[row, col]
                
                # Plot training data if available
                if metric_col in df.columns and len(df[metric_col]) > 0:
                    ax.plot(x, df[metric_col], color=color, linewidth=2.5, 
                           label=f'Training {title}', alpha=0.8)
                    
                    # Add intervention markers
                    self.add_intervention_markers(ax, x, df[metric_col], df, 
                                                intervention_col, stage_col)
                
                # Plot validation data if available and relevant
                if (df_val is not None and title == 'Mean Episode Reward' and 
                    'validation_avg_reward' in df_val.columns):
                    x_val = df_val['timestep']
                    ax.plot(x_val, df_val['validation_avg_reward'], 
                           color='black', linestyle='--', linewidth=2, 
                           label='Validation Reward', alpha=0.8)
                    ax.scatter(x_val, df_val['validation_avg_reward'], 
                             color='black', s=15, alpha=0.6, zorder=5)
                
                # Special handling for success rate on reward plot
                if (title == 'Mean Episode Reward' and df_val is not None and 
                    'validation_success_rate' in df_val.columns):
                    ax2 = ax.twinx()
                    ax2.plot(x_val, df_val['validation_success_rate'], 
                            color='red', linestyle=':', linewidth=2, 
                            label='Success Rate', alpha=0.7)
                    ax2.set_ylabel('Success Rate', color='red', fontsize=11)
                    ax2.tick_params(axis='y', labelcolor='red')
                    ax2.set_ylim(0, 1)
                    
                    # Combine legends
                    lines1, labels1 = ax.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                else:
                    # only show legend if there are labeled artists on the plot
                    handles, labels = ax.get_legend_handles_labels()
                    if handles:
                        ax.legend(fontsize=10)
                
                # Formatting
                ax.set_xlabel('Cumulative Iterations', fontsize=11)
                ax.set_ylabel(title, fontsize=11)
                ax.set_title(f'{title} Progress', fontsize=13, pad=10)
                ax.grid(True, alpha=0.3)
                
                # Improve tick formatting
                ax.tick_params(axis='both', which='major', labelsize=9)
            
            plt.tight_layout(pad=3.0, w_pad=2.5, h_pad=3.0)
            
            # Save plot with high DPI
            plot_path = os.path.join(self.output_dir, f'{heuristic_name}_comprehensive_analysis.png')
            plt.savefig(plot_path, dpi=600, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logging.info(f"Comprehensive analysis saved to {plot_path}")
            
        except Exception as e:
            logging.error(f"Error creating single heuristic analysis for {heuristic_name}: {e}")

    def plot_all_heuristics_comparison(self, heuristic_paths: Dict[str, str], 
                                     metric_keys: Optional[List[str]] = None):
        """Plot comparison of all heuristics on the same axes."""
        if metric_keys is None:
            metric_keys = [metric[1] for metric in self.training_metrics]
        
        # Load and process all data
        heuristic_data = {}
        for heuristic_name, path in heuristic_paths.items():
            try:
                df = pd.read_csv(path)
                df = self.add_cumulative_iteration(df)
                heuristic_data[heuristic_name] = df
            except Exception as e:
                logging.warning(f"Could not load data for {heuristic_name}: {e}")
        
        if not heuristic_data:
            logging.error("No valid heuristic data loaded")
            return
        
        # Create plots for each metric
        for metric_key in metric_keys:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Find the metric title
            metric_title = metric_key.replace('_', ' ').title()
            for title, col, _ in self.training_metrics:
                if col == metric_key:
                    metric_title = title
                    break
            
            fig.suptitle(f'{metric_title} Comparison Across All Heuristics', 
                        fontsize=18, y=0.95)
            
            # Plot each heuristic
            for heuristic_name, df in heuristic_data.items():
                if metric_key in df.columns:
                    x = df['cumulative_iteration']
                    y = df[metric_key]
                    
                    # Get color and label
                    color = self.colors.get(heuristic_name, 'gray')
                    label = 'Causal Mismatch' if heuristic_name == 'cm' else heuristic_name.title()
                    
                    ax.plot(x, y, color=color, linewidth=2.5, label=label, alpha=0.8)
                    
                    # Add intervention markers (lighter for multi-heuristic plot)
                    if 'stage' in df.columns and 'cumulative_iteration' in df.columns:
                        prev_stage = None
                        for idx, row in df.iterrows():
                            if idx == 0 or row['stage'] != prev_stage:
                                ax.axvline(x=row['cumulative_iteration'], 
                                         color=color, linestyle='--', alpha=0.2, linewidth=1)
                                prev_stage = row['stage']
            
            # Formatting
            ax.set_xlabel('Cumulative Iterations', fontsize=12)
            ax.set_ylabel(metric_title, fontsize=12)
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.legend(fontsize=11, loc='best')
            ax.grid(True, alpha=0.3)
            
            # Add subtle background
            ax.set_facecolor('#FAFAFA')
            
            plt.tight_layout()
            
            # Save plot
            filename = f'{metric_key}_comparison_all_heuristics.png'
            save_path = os.path.join(self.output_dir, filename)
            plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logging.info(f"Comparison plot saved: {save_path}")

    def create_modern_radar_plots(self, eval_data_paths: Dict[str, str]):
        """Create modern aesthetic radar plots for heuristic evaluation comparison."""
        
        # Load evaluation data
        heuristic_eval_data = {}
        for name, path in eval_data_paths.items():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    heuristic_eval_data[name] = data.get("final_evals", {})
            except Exception as e:
                logging.warning(f"Could not load evaluation data for {name}: {e}")
        
        if not heuristic_eval_data:
            logging.error("No valid evaluation data loaded for radar plots")
            return
        
        # Get protocol labels (assuming all files have same structure)
        first_heuristic = list(heuristic_eval_data.values())[0]
        protocol_labels = sorted(first_heuristic.keys(), key=lambda x: int(x[1:]))
        experiment_labels = list(heuristic_eval_data.keys())
        
        # Structure data for radar plots
        def extract_means_stds(eval_dict, protocol_keys, metric_key):
            """Extract means and stds for a given metric across all protocols."""
            means = [eval_dict[p_key].get(metric_key, 0) for p_key in protocol_keys]
            std_dev_key = 'std_' + metric_key[5:]  # Remove 'mean_' prefix
            stds = [eval_dict[p_key].get(std_dev_key, 0) for p_key in protocol_keys]
            return means, stds
        
        # Organize data
        radar_data_metrics = {}
        for metric in self.radar_metrics:
            radar_data_metrics[metric] = {}
            for heuristic, eval_data in heuristic_eval_data.items():
                means, stds = extract_means_stds(eval_data, protocol_labels, metric)
                radar_data_metrics[metric][heuristic] = (means, stds)
        
        # Create radar plots
        for metric_idx, metric_label in enumerate(self.radar_metrics):
            fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
            fig.patch.set_facecolor('white')
            
            # Calculate angles for radar chart
            num_vars = len(protocol_labels)
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            # Configure the radar chart aesthetics
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            
            # Set protocol labels
            ax.set_thetagrids(np.degrees(angles[:-1]), protocol_labels, 
                            fontsize=13, weight='bold')
            
            # Configure radial grid
            ax.set_ylim(0, 1.0)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], 
                             fontsize=11, color='#666666')
            ax.set_rlabel_position(180)
            
            # Style the grid
            ax.grid(True, color='#E0E0E0', linewidth=1.5, alpha=0.7)
            ax.set_facecolor('#FAFAFA')
            
            # Plot each heuristic
            for exp_idx, experiment in enumerate(experiment_labels):
                if experiment in radar_data_metrics[metric_label]:
                    values, _ = radar_data_metrics[metric_label][experiment]
                    values_plot = values + values[:1]  # Complete the circle
                    
                    # Get color and label
                    color = self.colors.get(experiment, '#333333')
                    label = 'Causal Mismatch' if experiment == 'cm' else experiment.title()
                    
                    # Plot line and fill
                    ax.plot(angles, values_plot, color=color, linewidth=3, 
                           label=label, marker='o', markersize=8, alpha=0.9)
                    ax.fill(angles, values_plot, color=color, alpha=0.2)
            
            # Enhanced title
            title_clean = metric_label.replace('_', ' ').replace('mean ', '').title()
            ax.set_title(title_clean, size=20, weight='bold', pad=40, color='#2C3E50')
            
            # Enhanced legend
            legend = ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
                             fontsize=14, frameon=True, fancybox=True, 
                             shadow=True, ncol=1, borderpad=1)
            legend.get_frame().set_facecolor('#FFFFFF')
            legend.get_frame().set_edgecolor('#CCCCCC')
            legend.get_frame().set_linewidth(1)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)
            
            # Save plot
            filename = f'radar_plot_{metric_label.lower().replace(" ", "_")}.png'
            save_path = os.path.join(self.output_dir, filename)
            plt.savefig(save_path, dpi=600, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            logging.info(f"Radar plot saved: {save_path}")

    def calculate_performance_ranking(self, eval_data_paths: Dict[str, str]) -> Dict[str, float]:
        """Calculate overall performance ranking based on mean_full_integrated_fractional_success."""
        
        # Load evaluation data
        heuristic_eval_data = {}
        for name, path in eval_data_paths.items():
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    heuristic_eval_data[name] = data.get("final_evals", {})
            except Exception as e:
                logging.warning(f"Could not load evaluation data for {name}: {e}")
        
        if not heuristic_eval_data:
            return {}
        
        # Calculate overall scores
        metric_key = 'mean_full_integrated_fractional_success'
        overall_scores = {}
        
        for heuristic, eval_data in heuristic_eval_data.items():
            scores = []
            for protocol, metrics in eval_data.items():
                if metric_key in metrics:
                    scores.append(metrics[metric_key])
            
            if scores:
                overall_scores[heuristic] = np.mean(scores)
        
        # Sort and display ranking
        ranked_heuristics = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        
        print("\n" + "="*50)
        print("HEURISTIC PERFORMANCE RANKING")
        print("="*50)
        for rank, (heuristic, score) in enumerate(ranked_heuristics, 1):
            display_name = 'Causal Mismatch' if heuristic == 'cm' else heuristic.title()
            print(f"{rank}. {display_name:<15}: {score:.4f}")
        
        if ranked_heuristics:
            best_heuristic = ranked_heuristics[0][0]
            best_name = 'Causal Mismatch' if best_heuristic == 'cm' else best_heuristic.title()
            print(f"\n🏆 Best performing heuristic: {best_name}")
            print(f"📊 Best score: {ranked_heuristics[0][1]:.4f}")
        
        return overall_scores

    def generate_comprehensive_report(self, base_paths: Dict[str, str], 
                                    eval_paths: Optional[Dict[str, str]] = None):
        """Generate a comprehensive visualization report."""
        
        logging.info("🎨 Generating comprehensive visualization report...")
        
        # 1. Individual heuristic analysis
        print("\n📈 Creating individual heuristic analyses...")
        for heuristic, base_path in base_paths.items():
            progress_path = os.path.join(base_path, 'all_progress.csv')
            validation_path = os.path.join(base_path, 'validation_log.csv')
            
            if os.path.exists(progress_path):
                self.plot_single_heuristic_analysis(progress_path, heuristic, 
                                                   validation_path if os.path.exists(validation_path) else None)
        
        # 2. Comparative analysis
        print("\n📊 Creating comparative analyses...")
        valid_progress_paths = {}
        for heuristic, base_path in base_paths.items():
            progress_path = os.path.join(base_path, 'all_progress.csv')
            if os.path.exists(progress_path):
                valid_progress_paths[heuristic] = progress_path
        
        if valid_progress_paths:
            self.plot_all_heuristics_comparison(valid_progress_paths)
        
        # 3. Radar plots and performance ranking
        if eval_paths:
            print("\n🎯 Creating radar plots and performance ranking...")
            valid_eval_paths = {k: v for k, v in eval_paths.items() if os.path.exists(v)}
            
            if valid_eval_paths:
                self.create_modern_radar_plots(valid_eval_paths)
                self.calculate_performance_ranking(valid_eval_paths)
        
        print(f"\n✅ Comprehensive visualization report completed!")
        print(f"📁 All plots saved to: {self.output_dir}")
    
    def load_all_log_files(self, base_path: str) -> dict:
        """Load all log files for a given heuristic into a dictionary of dataframes."""
        log_data = {}
        
        # Training log (episode data)
        training_log_path = os.path.join(base_path, 'training_log.csv')
        if os.path.exists(training_log_path):
            log_data['training'] = pd.read_csv(training_log_path)
            logging.info(f"Loaded training log with {len(log_data['training'])} episodes")
        
        # Validation log
        validation_log_path = os.path.join(base_path, 'validation_log.csv')
        if os.path.exists(validation_log_path):
            log_data['validation'] = pd.read_csv(validation_log_path)
            logging.info(f"Loaded validation log with {len(log_data['validation'])} validation points")
        
        # Intervention log
        intervention_log_path = os.path.join(base_path, 'intervention_log.csv')
        if os.path.exists(intervention_log_path):
            log_data['intervention'] = pd.read_csv(intervention_log_path)
            logging.info(f"Loaded intervention log with {len(log_data['intervention'])} intervention tests")
        
        # SB3 progress data
        progress_path = os.path.join(base_path, 'all_progress.csv')
        if os.path.exists(progress_path):
            log_data['progress'] = pd.read_csv(progress_path)
            logging.info(f"Loaded progress data with {len(log_data['progress'])} entries")
        
        # Sequencing results if available
        results_path = os.path.join(base_path, 'sequencing_results.json')
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                log_data['sequence'] = json.load(f)
            logging.info("Loaded sequencing results")
        
        return log_data
    
    def plot_curriculum_progression(self, heuristic_name: str, log_data: dict):
        """Plot the curriculum progression showing which interventions were selected."""
        if 'sequence' not in log_data or not log_data['sequence'].get('sequence'):
            logging.warning(f"No sequencing data available for curriculum progression plot for {heuristic_name}")
            return
        
        sequence = log_data['sequence']['sequence']
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(15, 7))
        
        # Extract data from sequence
        stages = [item['stage'] for item in sequence]
        interventions = [item['intervention'] for item in sequence]
        rewards = [item['test_metrics']['avg_reward'] for item in sequence]
        
        # Get success rates from intervention log by matching stage and intervention type
        success_rates = []
        if 'intervention' in log_data:
            intervention_df = log_data['intervention']
            for stage, intervention in zip(stages, interventions):
                # Find the row where this intervention was selected for this stage
                matching_rows = intervention_df[
                    (intervention_df['stage'] == stage) & 
                    (intervention_df['intervention_type'] == intervention) &
                    (intervention_df['selected'] == True)
                ]
                
                if not matching_rows.empty:
                    success_rates.append(matching_rows.iloc[0]['test_success_rate'])
                else:
                    # Fallback: find any test of this intervention in this stage
                    fallback_rows = intervention_df[
                        (intervention_df['stage'] == stage) & 
                        (intervention_df['intervention_type'] == intervention)
                    ]
                    if not fallback_rows.empty:
                        success_rates.append(fallback_rows.iloc[0]['test_success_rate'])
                    else:
                        logging.warning(f"No success rate found for stage {stage}, intervention {intervention}")
                        success_rates.append(0)
        else:
            logging.warning("No intervention log available - using zero success rates")
            success_rates = [0] * len(stages)

        # Plot bars for rewards
        bar_width = 0.35
        x = np.arange(len(stages))
        ax.bar(x - bar_width/2, rewards, bar_width, label='Test Reward', color='cornflowerblue')
        
        # Create a twin axis for success rate
        ax2 = ax.twinx()
        ax2.bar(x + bar_width/2, success_rates, bar_width, label='Test Success Rate', color='lightcoral')
        
        # Add intervention labels on bars
        for i, (intervention, reward) in enumerate(zip(interventions, rewards)):
            ax.text(i - bar_width/2, reward / 2, intervention, ha='center', va='center', color='white', fontsize=10, fontweight='bold', rotation=90)

        # Labels and title
        ax.set_xlabel('Curriculum Stage', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Reward', fontsize=12, color='cornflowerblue', fontweight='bold')
        ax2.set_ylabel('Success Rate', fontsize=12, color='lightcoral', fontweight='bold')
        ax.set_title(f'{heuristic_name.replace("_", " ").title()} - Curriculum Progression', fontsize=16, fontweight='bold')
        
        # Set the tick labels
        ax.set_xticks(x)
        ax.set_xticklabels([f'Stage {s}' for s in stages])
        
        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax2.set_ylim(0, 1.05)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f'{heuristic_name}_curriculum_progression.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Curriculum progression plot saved to {save_path}")
    
    def plot_training_validation_comparison(self, heuristic_name: str, log_data: dict):
        """Plot training versus validation metrics over time for reward, success, and episode length."""
        if 'training' not in log_data or 'validation' not in log_data:
            logging.warning(f"Missing training or validation data for comparison plot for {heuristic_name}")
            return
        
        training_df = log_data['training']
        validation_df = log_data['validation']
        
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 18), sharex=True)
        
        # Calculate total timesteps for training data
        training_df['total_timesteps'] = training_df['cumulative_timesteps'] + training_df['timestep']
        
        # --- Plot 1: Reward ---
        ax1.plot(training_df['total_timesteps'], training_df['reward'], alpha=0.2, color='dodgerblue', label='Episodic Training Reward')
        ax1.plot(training_df['total_timesteps'], training_df['mean_reward_last_10'], linewidth=2, color='darkblue', label='Training Reward (10-ep MA)')
        ax1.scatter(validation_df['timestep'], validation_df['validation_avg_reward'], color='crimson', s=60, zorder=5, label='Validation Reward')
        ax1.plot(validation_df['timestep'], validation_df['validation_avg_reward'], color='crimson', linestyle='--', alpha=0.7)
        
        # --- Plot 2: Success Rate ---
        ax2.plot(training_df['total_timesteps'], training_df['success'], alpha=0.2, color='seagreen', label='Episodic Training Success')
        ax2.plot(training_df['total_timesteps'], training_df['success_rate_last_10'], linewidth=2, color='darkgreen', label='Training Success Rate (10-ep MA)')
        ax2.scatter(validation_df['timestep'], validation_df['validation_success_rate'], color='orangered', s=60, zorder=5, label='Validation Success Rate')
        ax2.plot(validation_df['timestep'], validation_df['validation_success_rate'], color='orangered', linestyle='--', alpha=0.7)

        # --- Plot 3: Episode Length ---
        ax3.plot(training_df['total_timesteps'], training_df['episode_length'], alpha=0.2, color='purple', label='Episodic Training Length')
        ax3.plot(training_df['total_timesteps'], training_df['episode_length'].rolling(10).mean(), linewidth=2, color='indigo', label='Training Length (10-ep MA)')
        ax3.scatter(validation_df['timestep'], validation_df['validation_avg_length'], color='gold', s=60, zorder=5, label='Validation Length')
        ax3.plot(validation_df['timestep'], validation_df['validation_avg_length'], color='gold', linestyle='--', alpha=0.7)

        # Add vertical lines and text for stage changes
        if 'sequence' in log_data and log_data['sequence'].get('sequence'):
            sequence = log_data['sequence']['sequence']
            stage_timesteps = training_df.groupby('stage')['total_timesteps'].min()
            for i, item in enumerate(sequence):
                stage_start = stage_timesteps.get(item['stage'])
                if stage_start is not None:
                    for ax in [ax1, ax2, ax3]:
                        ax.axvline(x=stage_start, color='gray', linestyle='--', alpha=0.8)
                    stage_text = f"Stage {item['stage']}: {item['intervention']}"
                    ax1.text(stage_start + 50, ax1.get_ylim()[1] * 0.95, stage_text, rotation=90, verticalalignment='top', fontsize=9, color='dimgray')

        # Formatting
        fig.suptitle(f'{heuristic_name.replace("_", " ").title()} - Training & Validation Performance', fontsize=18, fontweight='bold')
        
        ax1.set_ylabel('Reward', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, linestyle='--', alpha=0.6)

        ax2.set_ylabel('Success Rate', fontsize=12)
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend(loc='upper left')
        ax2.grid(True, linestyle='--', alpha=0.6)

        ax3.set_xlabel('Cumulative Timesteps', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Episode Length', fontsize=12)
        ax3.legend(loc='upper left')
        ax3.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = os.path.join(self.output_dir, f'{heuristic_name}_training_validation_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Training-validation comparison plot saved to {save_path}")

    def plot_intervention_effectiveness(self, heuristic_name: str, log_data: dict):
        """Plot the effectiveness of each intervention type based on test runs."""
        if 'intervention' not in log_data:
            logging.warning(f"No intervention data available for effectiveness plot for {heuristic_name}")
            return
        
        intervention_df = log_data['intervention']
        
        # Group by intervention type and calculate average metrics
        intervention_metrics = intervention_df.groupby('intervention_type').agg(
            avg_reward=('test_avg_reward', 'mean'),
            std_reward=('test_avg_reward', 'std'),
            avg_success=('test_success_rate', 'mean'),
            std_success=('test_success_rate', 'std'),
            avg_length=('test_avg_length', 'mean'),
            std_length=('test_avg_length', 'std'),
            times_selected=('selected', lambda x: x.sum())
        ).reset_index().sort_values('avg_reward', ascending=False)
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 18), sharex=True)
        fig.suptitle(f'{heuristic_name.replace("_", " ").title()} - Intervention Test Effectiveness', fontsize=18, fontweight='bold')
        
        x = np.arange(len(intervention_metrics))
        
        # Plot Reward
        ax1.bar(x, intervention_metrics['avg_reward'], yerr=intervention_metrics['std_reward'], capsize=5, color='skyblue')
        ax1.set_ylabel('Average Reward', fontsize=12)
        ax1.set_title('Intervention Test Rewards', fontsize=14)
        
        # Plot Success Rate
        ax2.bar(x, intervention_metrics['avg_success'], yerr=intervention_metrics['std_success'], capsize=5, color='lightgreen')
        ax2.set_ylabel('Success Rate', fontsize=12)
        ax2.set_title('Intervention Test Success Rates', fontsize=14)
        ax2.set_ylim(0, 1.05)

        # Plot Episode Length
        ax3.bar(x, intervention_metrics['avg_length'], yerr=intervention_metrics['std_length'], capsize=5, color='lightcoral')
        ax3.set_ylabel('Episode Length', fontsize=12)
        ax3.set_title('Intervention Test Episode Lengths', fontsize=14)

        # Add labels for times selected
        for ax in [ax1, ax2, ax3]:
            for i, row in intervention_metrics.iterrows():
                if row['times_selected'] > 0:
                    ax.text(i, 0, f" Sel {int(row['times_selected'])}x ", color='white', ha='center', va='bottom', backgroundcolor='black', alpha=0.7, fontsize=9, fontweight='bold')

        plt.xticks(x, intervention_metrics['intervention_type'], rotation=45, ha='right')
        plt.xlabel('Intervention Type', fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        
        save_path = os.path.join(self.output_dir, f'{heuristic_name}_intervention_effectiveness.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Intervention effectiveness plot saved to {save_path}")
    
    def generate_heuristic_report(self, heuristic_name: str, base_path: str):
        """Generate a comprehensive report for a single heuristic."""
        # Load all log files
        log_data = self.load_all_log_files(base_path)
        
        if not log_data:
            logging.error(f"No log data found for {heuristic_name}")
            return
        
        logging.info(f"Generating comprehensive report for {heuristic_name}...")
        
        # the output directory is already set correctly in the constructor
        # heuristic_dir = os.path.join(self.output_dir, heuristic_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate each plot type
        self.plot_curriculum_progression(heuristic_name, log_data)
        self.plot_training_validation_comparison(heuristic_name, log_data)
        self.plot_intervention_effectiveness(heuristic_name, log_data)
        
        # Also create the original analysis plot for compatibility
        if 'progress' in log_data:
            progress_path = os.path.join(base_path, 'all_progress.csv')
            validation_path = os.path.join(base_path, 'validation_log.csv') if 'validation' in log_data else None
            self.plot_single_heuristic_analysis(progress_path, heuristic_name, validation_path)
        
        logging.info(f"Completed report generation for {heuristic_name}")

def generate_comprehensive_visualizations(log_base_dir: str, heuristics: List[str] = None):
    """Generate comprehensive visualizations for all heuristics."""
    
    if heuristics is None:
        # Auto-detect available heuristics
        heuristics = []
        possible_heuristics = ['greedy', 'cm', 'none', 'random', 'rnd', 'count', 'lpm', 'info']
        
        for h in possible_heuristics:
            # Check both with and without replacement
            for pattern in [f'{h}_sequencing_logs', f'{h}_replacement_sequencing_logs']:
                path = os.path.join(log_base_dir, pattern)
                if os.path.exists(path) and os.path.isdir(path):
                    heuristics.append((h, path))
    else:
        # Use specified heuristics - FIXED to check both patterns
        heuristics_list = heuristics    # store the input list
        heuristics = []                 # prepare the output list

        for h in heuristics_list:
            found = False
            # check both naming patterns
            for pattern in [f'{h}_sequencing_logs', f'{h}_replacement_sequencing_logs']:
                path = os.path.join(log_base_dir, pattern)
                if os.path.exists(path) and os.path.isdir(path):
                    heuristics.append((h, path))
                    found = True
                    logging.info(f"Found log directory for {h}: {path}")
                    break
            if not found:
                logging.warning(f"Could not find log directory for heuristic: '{h}")
    
    if not heuristics:
        logging.error(f"No valid heuristic directories found in {log_base_dir}")
        return
    
    # Create visualizer
    output_dir = os.path.join(log_base_dir, 'comprehensive_visualizations', heuristics_list[0])
    visualizer = EnhancedTrainingVisualizer(log_base_dir, output_dir)
    
    # Generate reports for each heuristic
    for name, path in heuristics:
        if os.path.exists(path):
            visualizer.generate_heuristic_report(name, path)
    
    # Generate cross-heuristic comparisons
    # This uses the existing methods to compare metrics across heuristics
    visualizer.plot_all_heuristics_comparison({name: os.path.join(path, 'all_progress.csv') 
                                              for name, path in heuristics 
                                              if os.path.exists(os.path.join(path, 'all_progress.csv'))})
    
    # Add radar plots if evaluation data is available
    eval_paths = {name: os.path.join(path, 'benchmark_results.json') 
                 for name, path in heuristics 
                 if os.path.exists(os.path.join(path, 'benchmark_results.json'))}
    
    if eval_paths:
        visualizer.create_modern_radar_plots(eval_paths)
        visualizer.calculate_performance_ranking(eval_paths)
    
    return visualizer

# Convenience function for easy usage
def create_enhanced_visualizations(log_base_dir: str, heuristics: List[str] = None, 
                                 output_dir: Optional[str] = None):
    """
    Convenience function to create enhanced visualizations for all available heuristics.
    
    Args:
        log_base_dir: Base directory containing all heuristic log folders
        heuristics: List of heuristic names to process (defaults to common ones)
        output_dir: Output directory for visualizations
    """
    
    if heuristics is None:
        heuristics = ['greedy', 'cm', 'none', 'random', 'rnd', 'count']
    
    # Set up paths
    base_paths = {}
    eval_paths = {}
    
    for heuristic in heuristics:
        log_dir = os.path.join(log_base_dir, f'{heuristic}_sequencing_logs')
        if os.path.exists(log_dir):
            base_paths[heuristic] = log_dir
            
            # Check for evaluation results
            eval_file = os.path.join(log_dir, 'benchmark_results.json')
            if os.path.exists(eval_file):
                eval_paths[heuristic] = eval_file
    
    if not base_paths:
        logging.error(f"No valid heuristic directories found in {log_base_dir}")
        return
    
    # Create visualizer and generate report
    output_path = output_dir or os.path.join(log_base_dir, 'enhanced_visualizations')
    visualizer = EnhancedTrainingVisualizer(log_base_dir, output_path)
    visualizer.generate_comprehensive_report(base_paths, eval_paths if eval_paths else None)
    
    return visualizer
