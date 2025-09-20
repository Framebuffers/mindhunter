import StatisticalObject
import PlottableObject
import pandas as pd
import numpy as np
from scipy.stats import norm

class StatisticalUtilities:
    def get_psd(self, x):
        mu=x.mean()
        sigma=x.std()
        minimum=x.min()
        maximum=x.max()
        x = np.linspace(minimum, maximum)
        pdf = norm.pdf(x, loc=mu, scale=sigma)
        return(x,pdf)
    
    