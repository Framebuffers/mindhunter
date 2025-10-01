from ..core.analyzer import DataAnalyzer
import numpy as np

class DistributionAnalyzer:
    def __init__(self, analyzer: DataAnalyzer) -> None:
        self.da = analyzer
    
    def check_basic_normality(self, data):
        mean_val = np.mean(data)
        median_val = np.median(data)
        std_val = np.std(data)
        print(f"Mean: {mean_val:.3f}")
        print(f"Median: {median_val:.3f}")
        print(f"Mean close to median?: {abs(mean_val - median_val) < 0.1 * std_val}")
        
        within_1std = np.sum(np.abs(data - mean_val) <= std_val) / len(data)
        print(f"Percentage within 1 std: {within_1std*100:.1f}% (should be ~68%)")