from ..mindhunter import StatFrame
from typing import Literal
from scipy import stats 
from scipy.stats import norm
from scipy.stats import pearsonr
from typing import Tuple, Any

import pandas as pd
import numpy as np

class AnalyticalTools:
    def __init__(self, sf: StatFrame):
        self.da = sf
    
    def z_to_p_value(self,
                      z: float,
                      alternative: Literal['two-sided', 'less', 'greater'] = 'two-sided') -> float:

        match alternative:
            case 'less':
                return stats.norm.cdf(z).item() # type: ignore
            case 'greater':
                return stats.norm.sf(z).item() # type: ignore
            case 'two-sided':
                return 2 * stats.norm.sf(abs(z)).item() # type: ignore

    def t_to_p_value(self, t_stat: float, df: int, alternative: str) -> float:
        """Convert t-statistic to p-value."""
        match alternative.lower():
            case 'less':
                return stats.t.cdf(t_stat, df).item()
            case 'greater':
                return stats.t.sf(t_stat, df).item()
            case 'two-sided':
                return 2 * stats.t.sf(abs(t_stat), df).item()
            case _:
                raise ValueError(f"Unknown alternative: {alternative}")
    
    def cv(self, *columns) -> pd.Series:
        data = self.da._df if not columns else self.da._df[list(columns)]
        return data.std() / data.mean()

    def z_score_all(self) -> pd.DataFrame | None:
        numeric_data = self.da._df.select_dtypes(include=[np.number])
        
        if numeric_data is None:
            return None
        
        return (numeric_data - numeric_data.mean()) / numeric_data.std()
    
    def z_score(self, column: str) -> float | None:
        if self.da._df[column].dtype != np.number:
            return None
        return self.da._df[column].std() / self.da._df[column].mean()
    
    def psd(self, x) -> Tuple[float, Any]:
        mu=x.mean()
        sigma=x.std()
        minimum=x.min()
        maximum=x.max()
        x = np.linspace(minimum, maximum)
        pdf = norm.pdf(x, loc=mu, scale=sigma)
        return(x,pdf)
    
    def wilson_score(self, p_hat: float, n: int, 
                                    alpha: float) -> tuple[float, float]:

        z = stats.norm.ppf(1 - alpha/2)
        denominator = 1 + z**2/n
        center = (p_hat + z**2/(2*n)) / denominator
        margin = z * np.sqrt(p_hat*(1-p_hat)/n + z**2/(4*n**2)) / denominator
        
        return max(0, center - margin), min(1, center + margin) # type: ignore
    
    def pearson_test(self, x: pd.Series, y: pd.Series) -> tuple[float, float]:
        if len(x) != len(y):
            raise ValueError("Series must have equal length")
        return pearsonr(x, y)
