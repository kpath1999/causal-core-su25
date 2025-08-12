#!/usr/bin/env python3
"""
Enhanced Visualization Generator Script

This script generates comprehensive visualizations for curriculum learning experiments.
It automatically detects available heuristic results and creates comparative analyses.

Usage:
    python generate_visualizations.py [--base_dir PATH] [--output_dir PATH] [--heuristics LIST]

Example:
    python generate_visualizations.py --base_dir /home/user/causal-core-su25 --heuristics greedy,cm,none,random
"""

import os
import sys
import argparse
import logging
from typing import List, Optional

# Add the source directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

try:
    from src.visualization import create_enhanced_visualizations
except ImportError:
    print("❌ Error: Cannot import visualization module.")
    print("Make sure the src/visualization.py file exists and is properly configured.")
    sys.exit(1)

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('visualization_generation.log')
        ]
    )

def detect_available_heuristics(base_dir: str) -> List[str]:
    """Automatically detect available heuristic directories."""
    heuristics = []
    possible_heuristics = ['greedy', 'cm', 'none', 'random', 'rnd', 'count']
    
    for heuristic in possible_heuristics:
        log_dir = os.path.join(base_dir, f'{heuristic}_sequencing_logs')
        if os.path.exists(log_dir):
            # Check if it has the required data files
            progress_file = os.path.join(log_dir, 'all_progress.csv')
            if os.path.exists(progress_file):
                heuristics.append(heuristic)
                logging.info(f"✓ Found data for {heuristic} heuristic")
            else:
                logging.warning(f"⚠️  Directory exists for {heuristic} but no progress data found")
        else:
            logging.info(f"ℹ️  No directory found for {heuristic} heuristic")
    
    return heuristics

def validate_paths(base_dir: str, output_dir: Optional[str] = None) -> tuple:
    """Validate input and output paths."""
    
    # Validate base directory
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    
    if not os.path.isdir(base_dir):
        raise NotADirectoryError(f"Base path is not a directory: {base_dir}")
    
    # Set up output directory
    if output_dir is None:
        output_dir = os.path.join(base_dir, 'enhanced_visualizations')
    
    os.makedirs(output_dir, exist_ok=True)
    
    return base_dir, output_dir

def main():
    """Main function to generate enhanced visualizations."""
    parser = argparse.ArgumentParser(
        description="Generate enhanced visualizations for curriculum learning experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_visualizations.py
  python generate_visualizations.py --base_dir /path/to/experiments
  python generate_visualizations.py --heuristics greedy,cm,none --output_dir /path/to/output
        """
    )
    
    parser.add_argument(
        '--base_dir', 
        type=str, 
        default='.',
        help='Base directory containing heuristic log folders (default: current directory)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Output directory for visualizations (default: base_dir/enhanced_visualizations)'
    )
    
    parser.add_argument(
        '--heuristics',
        type=str,
        help='Comma-separated list of heuristics to process (default: auto-detect)'
    )
    
    parser.add_argument(
        '--include_radar',
        action='store_true',
        help='Include radar plots (requires benchmark evaluation results)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    setup_logging()
    
    try:
        # Validate paths
        base_dir, output_dir = validate_paths(args.base_dir, args.output_dir)
        
        logging.info("🚀 Starting enhanced visualization generation...")
        logging.info(f"📁 Base directory: {base_dir}")
        logging.info(f"📊 Output directory: {output_dir}")
        
        # Determine heuristics to process
        if args.heuristics:
            heuristics = [h.strip() for h in args.heuristics.split(',')]
            logging.info(f"🎯 Using specified heuristics: {heuristics}")
        else:
            heuristics = detect_available_heuristics(base_dir)
            if not heuristics:
                logging.error("❌ No valid heuristic data found in the base directory")
                print("\n💡 Make sure your directory structure looks like:")
                print("   base_dir/")
                print("   ├── greedy_sequencing_logs/")
                print("   │   └── all_progress.csv")
                print("   ├── cm_sequencing_logs/")
                print("   │   └── all_progress.csv")
                print("   └── ...")
                sys.exit(1)
            
            logging.info(f"🔍 Auto-detected heuristics: {heuristics}")
        
        # Generate visualizations
        visualizer = create_enhanced_visualizations(
            log_base_dir=base_dir,
            heuristics=heuristics,
            output_dir=output_dir
        )
        
        # Print summary
        print("\n" + "="*60)
        print("📈 VISUALIZATION GENERATION COMPLETE")
        print("="*60)
        print(f"📁 Output location: {output_dir}")
        print(f"🎯 Processed heuristics: {', '.join(heuristics)}")
        
        # List generated files
        if os.path.exists(output_dir):
            files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
            if files:
                print(f"📊 Generated {len(files)} visualization files:")
                for file in sorted(files)[:10]:  # Show first 10
                    print(f"   • {file}")
                if len(files) > 10:
                    print(f"   ... and {len(files) - 10} more files")
            else:
                print("⚠️  No PNG files found in output directory")
        
        print("\n✅ Visualization generation completed successfully!")
        
    except Exception as e:
        logging.error(f"❌ Error during visualization generation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
