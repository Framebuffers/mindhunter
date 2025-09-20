import StatisticalObject
import PlottableObject
import pandas as pd
import numpy as np
from scipy.stats import norm

class StatisticalUtilities:
    """ Utilities to calculate statistical values for individual variables.
    
    Singleton class with access to commonly-used calculations for statistical values.
    These methods act only on the variables passed as arguments.
    
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, value):
        if not hasattr(self, '_initialized'):
            self.value = value
            self._initialized = True
    
    def get_psd(self, x):
        mu=x.mean()
        sigma=x.std()
        minimum=x.min()
        maximum=x.max()
        x = np.linspace(minimum, maximum)
        pdf = norm.pdf(x, loc=mu, scale=sigma)
        return(x,pdf)
    
    