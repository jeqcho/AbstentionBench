"""
Analysis package for AbstentionBench.

Provides utilities for loading results and generating tables.
"""

from .load_results import Results
from .tables import AbstentionF1ScoreTable

__all__ = ["Results", "AbstentionF1ScoreTable"]