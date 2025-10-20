"""

mindhunter
Statistical Analysis Extensions for Pandas DataFrames

"""
# core
from .mindhunter import StatFrame
from .analyzer import StatAnalyzer
from .modeller import StatModel
from .plotter import StatPlotter
from .tools import StatTools
from .visualizer import StatVisualizer
from .database import StatDatabase

__version__ = '0.1.2'
__name__ = 'mindhunter'
__all__ = [
    'StatFrame',
    'StatAnalyzer',
    'StatModel',
    'StatPlotter',
    'StatTools',
    'StatDatabase',
    'StatVisualizer'
]