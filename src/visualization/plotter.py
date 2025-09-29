from core.analyzer import DataAnalyzer
from typing import List
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

class Visualizer:
    def __init__(self, analyzer: DataAnalyzer):
        self.da = analyzer
    
    def create_scatterplot(self, columns: List[str]) -> None:
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                col1 = columns[i]
                col2 = columns[j]
                if col1 in self.da.df.columns and col2 in self.da.df.columns:
                    plt.figure(figsize=(8, 6))
                    sns.scatterplot(data=self.da.df, x=col1, y=col2)
                    plt.title(f'Scatterplot de {col1} vs {col2}')
                    plt.xlabel(col1)
                    plt.ylabel(col2)
                    plt.show()
    
    def create_boxplot(self, columns: List[str]) -> None:
        for col in columns:
            if col in self.da.df.columns:
                plt.figure(figsize=(8, 6))
                sns.boxplot(y=self.da.df[col])
                plt.title(f'Boxplot de {col}')
                plt.ylabel(col)
                plt.show()
                
