"""

mindhunter
Statistical Analysis Extensions for Pandas DataFrames

"""
# core
from .mindhunter import StatFrame

# statistics
from .statistics.distributions import DistributionAnalyzer
from .statistics.hypothesis import HypothesisAnalyzer

# utils
from .utils.toolkit import AnalyticalTools

# visualization
from .visualization.stat_plotter import StatPlotter
from .visualization.visualizer import StatVisualizer

__version__ = '0.1.0'
__name__ = 'mindhunter'
__all__ = [
    'StatFrame',
    'DistributionAnalyzer',
    'HypothesisAnalyzer',
    'AnalyticalTools',
    'StatPlotter',
    'StatVisualizer',
]