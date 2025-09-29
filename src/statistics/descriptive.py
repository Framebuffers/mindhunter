from core.analyzer import DataAnalyzer
import pandas as pd
import numpy as np

class DescriptiveStats:
    def __init__(self, analyzer: DataAnalyzer):
        self.da = analyzer
    
