import pandas as pd
import numpy as np
import seaborn as sns # type: ignore
import re
import matplotlib.pyplot as plt
import scipy as sp
from scipy.stats import norm
from scipy import stats
from .mindhunter import StatFrame
from typing import List, Literal, Tuple, Any
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, mean_squared_error, r2_score, roc_curve

class StatAnalyzer:
    def __init__(self, sf: StatFrame):
        self.da = sf
        # self.tools = AnalyticalTools(sf)
    pass