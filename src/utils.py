"""
Utility functions for data processing and analysis.
"""

import os
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional, Tuple
import logging

def load_progress_data(log_dir: str) -> Optional[pd.DataFrame]:
    """Load and validate progress data from a log directory."""
    progress_path = os.path.join(log_dir, 'all_progress.csv')
    
    if not os.path.exists(progress_path):
        logging.warning(f"Progress file not found: {progress_path}")
        return None
    
    try:
        df = pd.read_csv(progress_path)
        
        # Validate required columns
        required_cols = ['iteration', 'stage', 'mean_episode_reward']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logging.error(f"Missing required columns in {progress_path}: {missing_cols}")
            return None
        
        return df
        
    except Exception as e:
        logging.error(f"Error loading progress data from {progress_path}: {e}")
        return None

def load_validation_data(log_dir: str) -> Optional[pd.DataFrame]:
    """Load validation data if available."""
    validation_path = os.path.join(log_dir, 'validation_log.csv')
    
    if not os.path.exists(validation_path):
        return None
    
    try:
        return pd.read_csv(validation_path)
    except Exception as e:
        logging.warning(f"Error loading validation data from {validation_path}: {e}")
        return None

def load_evaluation_results(log_dir: str) -> Optional[Dict]:
    """Load benchmark evaluation results."""
    eval_path = os.path.join(log_dir, 'benchmark_results.json')
    
    if not os.path.exists(eval_path):
        return None
    
    try:
        with open(eval_path, 'r') as f:
            data = json.load(f)
        return data.get('final_evals', {})
    except Exception as e:
        logging.warning(f"Error loading evaluation results from {eval_path}: {e}")
        return None

def get_heuristic_summary(base_dir: str) -> Dict[str, Dict]:
    """Get a summary of available data for each heuristic."""
    summary = {}
    possible_heuristics = ['greedy', 'cm', 'none', 'random', 'rnd', 'count']
    
    for heuristic in possible_heuristics:
        log_dir = os.path.join(base_dir, f'{heuristic}_sequencing_logs')
        
        if not os.path.exists(log_dir):
            continue
        
        heuristic_info = {
            'log_dir': log_dir,
            'has_progress': False,
            'has_validation': False,
            'has_evaluation': False,
            'progress_episodes': 0,
            'validation_episodes': 0
        }
        
        # Check progress data
        progress_df = load_progress_data(log_dir)
        if progress_df is not None:
            heuristic_info['has_progress'] = True
            heuristic_info['progress_episodes'] = len(progress_df)
        
        # Check validation data
        validation_df = load_validation_data(log_dir)
        if validation_df is not None:
            heuristic_info['has_validation'] = True
            heuristic_info['validation_episodes'] = len(validation_df)
        
        # Check evaluation results
        eval_results = load_evaluation_results(log_dir)
        if eval_results:
            heuristic_info['has_evaluation'] = True
        
        summary[heuristic] = heuristic_info
    
    return summary

def calculate_training_statistics(df: pd.DataFrame) -> Dict:
    """Calculate basic statistics from training data."""
    stats = {}
    
    if 'mean_episode_reward' in df.columns:
        rewards = df['mean_episode_reward'].dropna()
        stats['reward'] = {
            'final': rewards.iloc[-1] if len(rewards) > 0 else 0,
            'max': rewards.max() if len(rewards) > 0 else 0,
            'mean': rewards.mean() if len(rewards) > 0 else 0,
            'std': rewards.std() if len(rewards) > 0 else 0
        }
    
    if 'mean_episode_length' in df.columns:
        lengths = df['mean_episode_length'].dropna()
        stats['episode_length'] = {
            'final': lengths.iloc[-1] if len(lengths) > 0 else 0,
            'mean': lengths.mean() if len(lengths) > 0 else 0
        }
    
    if 'stage' in df.columns:
        stats['stages_completed'] = df['stage'].max() if len(df) > 0 else 0
    
    return stats

def create_summary_report(base_dir: str, output_file: Optional[str] = None) -> str:
    """Create a text summary report of all available data."""
    
    summary = get_heuristic_summary(base_dir)
    
    report_lines = [
        "="*60,
        "CURRICULUM LEARNING EXPERIMENTS SUMMARY",
        "="*60,
        f"Base directory: {base_dir}",
        f"Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "AVAILABLE HEURISTICS:",
        "-"*30
    ]
    
    if not summary:
        report_lines.append("❌ No valid heuristic data found")
    else:
        for heuristic, info in summary.items():
            status_icons = []
            if info['has_progress']:
                status_icons.append("📈")
            if info['has_validation']:
                status_icons.append("✅")
            if info['has_evaluation']:
                status_icons.append("🎯")
            
            status_str = " ".join(status_icons) if status_icons else "❌"
            
            report_lines.append(f"{heuristic.upper():<12} {status_str}")
            if info['has_progress']:
                report_lines.append(f"             Training episodes: {info['progress_episodes']}")
            if info['has_validation']:
                report_lines.append(f"             Validation episodes: {info['validation_episodes']}")
            report_lines.append("")
    
    # Add training statistics if available
    report_lines.extend([
        "TRAINING STATISTICS:",
        "-"*30
    ])
    
    for heuristic, info in summary.items():
        if not info['has_progress']:
            continue
        
        progress_df = load_progress_data(info['log_dir'])
        if progress_df is not None:
            stats = calculate_training_statistics(progress_df)
            
            report_lines.append(f"{heuristic.upper()}")
            if 'reward' in stats:
                r = stats['reward']
                report_lines.append(f"  Final reward: {r['final']:.3f}")
                report_lines.append(f"  Max reward:   {r['max']:.3f}")
                report_lines.append(f"  Mean reward:  {r['mean']:.3f} ± {r['std']:.3f}")
            
            if 'stages_completed' in stats:
                report_lines.append(f"  Stages:       {stats['stages_completed']}")
            
            report_lines.append("")
    
    report_lines.extend([
        "LEGEND:",
        "📈 = Training data available",
        "✅ = Validation data available", 
        "🎯 = Evaluation results available",
        "="*60
    ])
    
    report_text = "\n".join(report_lines)
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"Summary report saved to: {output_file}")
    
    return report_text

# Convenience function for quick analysis
def quick_analysis(base_dir: str = '.'):
    """Perform a quick analysis and print summary."""
    print(create_summary_report(base_dir))
