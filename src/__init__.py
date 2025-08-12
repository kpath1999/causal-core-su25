"""
Source package for enhanced curriculum learning analysis and visualization.
"""

try:
    from .visualization import EnhancedTrainingVisualizer, create_enhanced_visualizations
    from .utils import quick_analysis, create_summary_report, get_heuristic_summary
    
    __all__ = [
        'EnhancedTrainingVisualizer',
        'create_enhanced_visualizations', 
        'quick_analysis',
        'create_summary_report',
        'get_heuristic_summary'
    ]
    
except ImportError as e:
    import warnings
    warnings.warn(f"Some modules could not be imported: {e}")
    __all__ = []

__version__ = "1.0.0"
