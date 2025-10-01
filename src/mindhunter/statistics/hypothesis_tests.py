from ..core.analyzer import DataAnalyzer
from ..utils.toolkit import AnalysisToolkit
from typing import Literal
from scipy import stats
import scipy as sp
import pandas as pd
import numpy as np


class HypothesisTester:
    def __init__(self, analyzer: DataAnalyzer):
        self.da = analyzer
        self.tools = AnalysisToolkit(analyzer)
    
    def hypothesis_test(self, 
                        column: str,
                        test_type: str,
                        null_value: float,
                        alpha: float = 0.05,
                        alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided') -> dict:

        if column not in self.da._df.columns:
            raise ValueError(f"Column '{column}' not found")

        data = self.da._df[column].dropna()
        n = len(data)
        
        test_stat, p_value = self._perform_test(data, test_type, null_value, alternative)
        
        return {
            'test_statistic': test_stat,
            'p_value': p_value,
            'alpha': alpha,
            'reject_null': p_value < alpha,
            'sample_size': n,
            'test_type': test_type,
            'conclusion': 'Reject H0' if p_value < alpha else 'Could not reject H0'
        }

    def get_binomial_mean_comparison(self, column: str, n: int, p: float) -> dict[str, float]:

        if column not in self.da._df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        sample_mean = self.da._df[column].mean()
        theoretical_mean = n * p
        
        return {
            'sample_mean': sample_mean,
            'theoretical_mean': theoretical_mean,
            'difference': abs(sample_mean - theoretical_mean)
        }
        
    def _perform_test(self, 
                    data: pd.Series, 
                    test_type: str, 
                    null_value: float, 
                    alternative: str) -> tuple[float, float]:

        match test_type.lower():
            case 'one_sample_t':
                return stats.ttest_1samp(data, null_value, alternative=alternative)
            case 'z_test':
                z_stat = (data.mean() - null_value) / (data.std() / np.sqrt(len(data)))
                p_val = self._z_to_p_value(z_stat, alternative) # type: ignore
                return z_stat, p_val
            case 'binomial':
                successes = (data == 1).sum()
                binom_test = stats.binomtest(successes, len(data), null_value, alternative=alternative)
                return binom_test.statistic, np.float64(binom_test.pvalue)
            case _:
                raise ValueError(f"Unsupported test: {test_type}")
            
   
    def _proportion_test(self, successes: int, n: int, p0: float, 
                        alternative: str) -> tuple[float, float]:
        """ Calculates proportion using binomial tests. """

        if n < 30 or n * p0 < 5 or n * (1 - p0) < 5:
            p_value = stats.binom_test(successes, n, p0, alternative=alternative) # type: ignore
            z_stat = (successes/n - p0) / np.sqrt(p0 * (1-p0) / n)
            return z_stat, p_value
        
        sample_prop = successes / n
        se = np.sqrt(p0 * (1 - p0) / n)
        z_stat = (sample_prop - p0) / se
        p_value = self._z_to_p_value(z_stat, alternative) # type: ignore
        
        return z_stat, p_value

    def _interpret_correlation(self, r: float) -> str:
        abs_r = abs(r)
        if abs_r >= 0.7:
            strength = "Strong"
        elif abs_r >= 0.3:
            strength = "Moderate" 
        elif abs_r >= 0.1:
            strength = "Weak"
        else:
            strength = "Very weak"
        
        direction = "positive" if r > 0 else "negative" if r < 0 else "no"
        return f"{strength} {direction} correlation" 

