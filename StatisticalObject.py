import pandas as pd
import numpy as np
import seaborn as sns # type: ignore
import re
import matplotlib as plt
import scipy as sp
from scipy.stats import norm
from scipy import stats
from typing import List, Literal, Union

class StatisticalObject:
 
    
    def __init__(self, dataframe: pd.DataFrame) -> None:
        """ Wrapper for `DataFrame`, adding quick access to statistical values, tools and functions.
            
            Provides facilities to:
                - Clean a DataFrame
                - Obtain columns by passing names as arguments
                - Access essencial stats from a DataFrame
                
            To access plotting capabilities, refer to `PlottableObject`. 
            
        """
        self.statistical_object_df = dataframe.copy()
        self.df_stats = self.statistical_object_df.describe()
        self.df_columns = self.statistical_object_df.columns.to_list()
        self._cached_stats = {}
        self._compute_essential_stats()
    
    # creation methods
    def from_csv(self, csv) -> StatisticalObject: # type: ignore
        return StatisticalObject(pd.read_csv(csv))
    
    # data management methods
    def clean_df(self, *chars_to_remove) -> None:
        if chars_to_remove:
            escaped_chars = ''.join(re.escape(char) for char in chars_to_remove)
            pattern = f"[{escaped_chars}]"
        else:
            pattern = r"[^\w\s]"

        normalized_columns = [
            re.sub(pattern, '_', col.lower()).replace(' ', '_')
            for col in self.statistical_object_df.columns
        ]
        
        self.statistical_object_df.columns = normalized_columns
        self.statistical_object_df.dropna(inplace=True)
        self.statistical_object_df.drop_duplicates(inplace=True)
    
    def get_columns(self, *column_names) -> pd.DataFrame:
        if not column_names:
            print("No column names provided.")
            return pd.DataFrame() # Return an empty DataFrame if no columns are specified

        valid_columns = [col for col in column_names if col in self.df_columns]
        
        if len(valid_columns) != len(column_names):
            invalid_columns = set(column_names) - set(valid_columns)
            print(f"Warning: The following columns were not found in the DataFrame: {', '.join(invalid_columns)}")

        return self.statistical_object_df[list(valid_columns)]
   
    def get_df(self) ->  pd.DataFrame:
        return self.statistical_object_df
    
    def describe(self, *columns) -> pd.DataFrame:
        data = self.statistical_object_df if not columns else self.statistical_object_df[[*columns]]
        return data.describe()
   
    def _compute_essential_stats(self):
        """ Compute and cache essential statistical measures.
        
            Given the loaded `DataFrame`, it calculates a series of values:
            
            Central Tendency:
                - mean
                - median
                - mode
            
            Spread/Variability (for testing):
                - std (standard deviation)
                - variance
                - range
                - iqr (inter-quantile range)
                - mad (median absolute deviation)
            
            Distribution Shape (for normality assumptions):
                - skewness
                - kurtosis
                
            Data Quality:
                - count
                - missing_count
                - missing_pct
            
            Extreme Values (outliers):
                - min
                - max
                - q1
                - q3
            
            Key Ratios (for standardised measurements):
                - cv (coefficient of variation)
                - sem (standard error of mean)
        
        """
        numeric_cols = self.statistical_object_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            data = self.statistical_object_df[col].dropna()
            
            self._cached_stats[col] = {
                'mean': data.mean(),
                'median': data.median(),
                'mode': data.mode().iloc[0] if not data.mode().empty else np.nan,
                
                'std': data.std(),
                'variance': data.var(),
                'range': data.max() - data.min(),
                'iqr': data.quantile(0.75) - data.quantile(0.25),
                'mad': (data - data.median()).abs().median(),
                
                'skewness': data.skew(),
                'kurtosis': data.kurtosis(),
                
                'count': len(data),
                'missing_count': self.statistical_object_df[col].isna().sum(),
                'missing_pct': self.statistical_object_df[col].isna().mean(),
                
                'min': data.min(),
                'max': data.max(),
                'q1': data.quantile(0.25),
                'q3': data.quantile(0.75),
                
                'cv': data.std() / data.mean() if data.mean() != 0 else np.inf,
                'sem': data.std() / np.sqrt(len(data))
            }
            
    @property
    def essential_stats(self) -> dict:
        """ Get all cached essential statistics.

            Returns:
                A dictionary with essential statistical values for: central tendency, spread/variability, distribution shape, data quality, extreme values and key ratios. The values stored are:
                    - #Central Tendency:
                        - mean
                        - median
                        - mode
                    - #Spread/Variability (for testing):
                        - std #(standard deviation)
                        - variance
                        - range
                        - iqr #(inter-quantile range)
                        - mad #(median absolute deviation)
                    - #Distribution Shape (for normality assumptions):
                        - skewness
                        - kurtosis
                    - #Data Quality:
                        - count
                        - missing_count
                        - missing_pct
                    - #Extreme Values (outliers):
                        - min
                        - max
                        - q1
                        - q3
                    - #Key Ratios (for standardised measurements):
                        - cv #(coefficient of variation)
                        - sem #(standard error of mean)
        """
        
        return self._cached_stats
    
    def get_quick_stats(self, column: str) -> dict:
        """ Get essential stats for a specific column.
        
        """
        
        if column not in self._cached_stats:
            raise ValueError(f"Column '{column}' not found in cached stats")
        return self._cached_stats[column]
   
    def z_score(self) -> pd.DataFrame:
        numeric_data = self.statistical_object_df.select_dtypes(include=[np.number])
        return (numeric_data - numeric_data.mean()) / numeric_data.std()