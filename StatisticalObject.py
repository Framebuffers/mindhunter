import pandas as pd
import numpy as np
import seaborn as sns # type: ignore
import re
import matplotlib as plt
import scipy as sp
from scipy.stats import norm
from scipy import stats
from typing import List, Literal, Union

""" because I always for get how to do list comprehensions:

[expression for item in iterable if condition]
#    ↑         ↑        ↑           ↑
# Select    foreach   source    Where

"""

class StatisticalObject:
    
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.df = dataframe.copy()
        self.df_stats = self.df.describe()
        self.df_columns = self.df.columns.to_list()
        
    def from_csv(self, csv) -> StatisticalObject: # type: ignore
        return StatisticalObject(pd.read_csv(csv))
    
    def clean_df(self, *chars_to_remove) -> None:
        if chars_to_remove:
            escaped_chars = ''.join(re.escape(char) for char in chars_to_remove)
            pattern = f"[{escaped_chars}]"
        else:
            pattern = r"[^\w\s]"

        normalized_columns = [
            re.sub(pattern, '_', col.lower()).replace(' ', '_')
            for col in self.df.columns
        ]
        
        self.df.columns = normalized_columns
        self.df.dropna(inplace=True)
        self.df.drop_duplicates(inplace=True)
    
    def get_columns(self, *column_names) -> pd.DataFrame:
        if not column_names:
            print("No column names provided.")
            return pd.DataFrame() # Return an empty DataFrame if no columns are specified

        valid_columns = [col for col in column_names if col in self.df_columns]
        
        if len(valid_columns) != len(column_names):
            invalid_columns = set(column_names) - set(valid_columns)
            print(f"Warning: The following columns were not found in the DataFrame: {', '.join(invalid_columns)}")

        return self.df[list(valid_columns)]
   
    def get_df(self) ->  pd.DataFrame:
        return self.df
    
    def describe(self, *columns) -> pd.DataFrame:
        data = self.df if not columns else self.df[[*columns]]
        return data.describe()
    
    def get_coefficient_variation(self, *columns) -> pd.Series:
        data = self.df if not columns else self.df[list(columns)]
        return data.std() / data.mean() 
    
    def z_score(self) -> pd.DataFrame:
        numeric_data = self.df.select_dtypes(include=[np.number])
        return (numeric_data - numeric_data.mean()) / numeric_data.std()
    
    def get_psd(self, x):
        mu=x.mean()
        sigma=x.std()
        minimum=x.min()
        maximum=x.max()
        x = np.linspace(minimum, maximum)
        pdf = norm.pdf(x, loc=mu, scale=sigma)
        return(x,pdf)
    
    