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
