import pandas as pd
import numpy as np
import re

class StatFrame:
    def __init__(self, df: pd.DataFrame, precalc_data: bool = True):
        self._df = df.copy()
        self._cached_stats = {}
        self.df_stats = self.df.describe()
        self.df_columns = self.df.columns.to_list()
        if precalc_data == True:
            self._compute_essential_stats()
    
    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def update(self):
        self._compute_essential_stats()
    
    def clean_df(self, *chars_to_remove) -> None:
        if chars_to_remove:
            escaped_chars = ''.join(re.escape(char) for char in chars_to_remove)
            pattern = f"[{escaped_chars}]"
        else:
            pattern = r"[^\w\s]"

        normalized_columns = [
            re.sub(pattern, '_', col.lower()).replace(' ', '_')
            for col in self._df.columns
        ]
        
        self._df.columns = normalized_columns
        self._df.dropna(inplace=True)
        self._df.drop_duplicates(inplace=True)
    
    def locate_zero_rows(self, columns: list[str] = None,  # type: ignore
                    return_indices: bool = False) -> pd.DataFrame | list:

        if columns is None:
            check_columns = self._df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            check_columns = columns
        
        zero_mask = (self._df[check_columns] == 0).any(axis=1)
        zero_rows = self._df[zero_mask]
        
        if return_indices:
            return zero_rows.index.tolist()
        return zero_rows
    
    def analyze_zero_removal(self) -> pd.DataFrame:
        numeric_cols = self._df.select_dtypes(include=[np.number]).columns
        
        analysis = []
        for col in numeric_cols:
            zero_count = (self._df[col] == 0).sum()
            zero_pct = (zero_count / len(self._df)) * 100
            
            analysis.append({
                'column': col,
                'zero_count': zero_count,
                'zero_percentage': f"{zero_pct:.1f}%",
                'total_rows': len(self._df)
            })
        
        return pd.DataFrame(analysis)
    
    def remove_exact_zeros(self, update_cache: bool = True) -> dict:
        numeric_cols = self._df.select_dtypes(include=[np.number]).columns
        
        zero_mask = (self._df[numeric_cols] == 0.0).any(axis=1)
        
        original_length = len(self._df)
        self._df = self._df[~zero_mask].reset_index(drop=True)
        
        if update_cache:
            self._cached_stats = {}
            self._compute_essential_stats()
        
        return {
            'method': 'exact_zeros',
            'rows_removed': original_length - len(self._df),
            'new_length': len(self._df)
        }

    def remove_near_zeros(self, tolerance: float = 1e-10, 
                        columns: list[str] = None, update_cache: bool = True) -> dict: # type: ignore
        if columns is None:
            columns = self._df.select_dtypes(include=[np.number]).columns.tolist()
        
        near_zero_mask = (abs(self._df[columns]) <= tolerance).any(axis=1)
        
        original_length = len(self._df)
        self._df = self._df[~near_zero_mask].reset_index(drop=True)
        
        if update_cache:
            self._cached_stats = {}
            self._compute_essential_stats()
        
        return {
            'method': 'near_zeros',
            'tolerance_used': tolerance,
            'rows_removed': original_length - len(self._df),
            'columns_checked': columns
        }  
    
    def describe_columns(self, *columns: str) -> pd.DataFrame:
        return self._df[list(columns)].describe() if columns else self._df.describe()

    def get_stats(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self._cached_stats)
    
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
            numeric_cols = self._df.select_dtypes(include=[np.number]).columns

            for col in numeric_cols:
                data = self._df[col].dropna()
                
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
                    'missing_count': self._df[col].isna().sum(),
                    'missing_pct': self._df[col].isna().mean(),
                    
                    'min': data.min(),
                    'max': data.max(),
                    'q1': data.quantile(0.25),
                    'q3': data.quantile(0.75),
                    
                    'cv': data.std() / data.mean() if data.mean() != 0 else np.inf,
                    'sem': data.std() / np.sqrt(len(data))
                }

    def _compute_column_stats(self, column_name: str) -> None:
        data = self._df[column_name].dropna()

        self._df[column_name] = {
            'mean': data.mean(),
            'std': data.std(),
            'median': data.median(),
            'count': len(data),
            'missing_pct': self._df[column_name].isna().mean(),
            'min': data.min(),
            'max': data.max(),
            'cv': data.std() / data.mean() if data.mean() != 0 else np.inf,
            'sem': data.std() / np.sqrt(len(data)),
            'skewness': data.skew()
        }