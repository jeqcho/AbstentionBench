"""
Data files and utilities for AbstentionBench datasets.
"""
import os
from pathlib import Path
try:
    from importlib import resources
except ImportError:
    # Python < 3.9 fallback
    import importlib_resources as resources


def get_data_dir(subdir=None):
    """Get the path to the data directory, creating it if it doesn't exist.
    
    Args:
        subdir: Optional subdirectory within the data directory
        
    Returns:
        Path to the data directory or subdirectory
    """
    try:
        # When installed as a package, use the installed data directory
        with resources.path(__name__, '.') as data_path:
            base_dir = Path(data_path)
    except (ImportError, AttributeError, FileNotFoundError):
        # Fallback for development or when resources don't work
        base_dir = Path(__file__).parent
    
    if subdir:
        full_path = base_dir / subdir
        full_path.mkdir(parents=True, exist_ok=True)
        return str(full_path)
    else:
        return str(base_dir)