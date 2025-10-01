"""

mindhunter
Statistical Analysis Extensions for Pandas DataFrames

"""
# core
from .core.analyzer import DataAnalyzer

# statistics
from .statistics.distributions import DistributionAnalyzer
from .statistics.hypothesis_tests import HypothesisTester

# utils
from .utils.toolkit import AnalysisToolkit

# visualization
from .visualization.stat_plotter import StatisticalPlotter
from .visualization.plotter import Visualizer

__version__ = '0.1.0'
__name__ = 'mindhunter'
__all__ = [
    'DataAnalyzer',
    'DistributionAnalyzer',
    'HypothesisTester',
    'AnalysisToolkit',
    'StatisticalPlotter',
    'Visualizer',
]
