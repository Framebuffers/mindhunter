"""

mindhunter
Statistical Analysis Extensions for Pandas DataFrames

(C) 2025 - Sebastian Torres Sagredo (Framebuffer) 
Released under the AGPLv3 license.

"""

from .core.analyzer import DataAnalyzer
from .statistics.descriptive import DescriptiveStats
from .statistics.distributions import DistributionAnalyzer
from .statistics.hypothesis_tests import HypothesisTester
from .visualization.distributions import StatisticalPlotter
from .visualization.plotter import Visualizer
from .utils.toolkit import AnalysisToolkit

__version__ = '0.1.0'

__all__ = [
    'DataAnalyzer',
    'DescriptiveStats',
    'DistributionAnalyzer',
    'HypothesisTester',
    'StatisticalPlotter',
    'HypothesisTester',
    'Visualizer',
    'AnalysisToolkit'
]