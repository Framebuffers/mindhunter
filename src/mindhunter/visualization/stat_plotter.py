from ..mindhunter import StatFrame
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from scipy import stats
import scipy as sp

class StatPlotter:
    def __init__(self, sf: StatFrame):
        self.da = sf
    
    def plot_z_scores(self, *columns: str) -> None:
        if not columns:
            numeric_cols = self.da.df.select_dtypes(include=[np.number]).columns
            z_scores = self.da.df[numeric_cols]
        else:
            z_scores = self.da.df[[*columns]]
        
        z_scores = (z_scores - z_scores.mean()) / z_scores.std()
        z_melted = z_scores.melt(var_name='Variable', value_name='Z-score')
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=z_melted, x='Variable', y='Z-score')
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        plt.axhline(y=2, color='orange', linestyle='--', alpha=0.5)
        plt.axhline(y=-2, color='orange', linestyle='--', alpha=0.5)
        plt.title('Z-scores Across Variables')
        plt.xticks(rotation=45)
        plt.show()
    
    def plot_coefficient_variation(self, 
                                    title: str = 'Coefficient of Variation of Indicators', 
                                    x_label: str = 'Indicators', 
                                    y_label: str = 'Coefficient of Variation', 
                                    rotation: int = 90, 
                                    ha: str = 'right') -> None:
        """ Graphs the Coefficient of Variation for each column.

            The Coefficient of Variation is calculated as the standard deviation divided by the mean.
            It is graphed as a barplot with the 5 most volatile indicators.

            Args:
                overall (pd.DataFrame): DataFrame held within the StatisticalObject.
            Returns:
                None
            Raises:
                None
            
            """
        data_frame = self.da.df
        plt.figure(figsize=(10, 6))
        plt.title(title)
        sns.barplot(x=data_frame.columns, y=data_frame.std() / data_frame.mean(), width=.5)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.xticks(rotation=rotation, ha=ha)
        plt.show()

    def plot_normality_check(self, column_name: str) -> None:
        basic_stats = self.da._cached_stats
        data = self.da.df
        ndev=sns.histplot(data, bins=30, x=column_name, kde=True)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, loc=basic_stats['mean'], scale=basic_stats['std'])
        plt.plot(x, p, 'k', linewidth=2)
        title = "Fit results: mu = %.2f, std = %.2f" % (basic_stats['mean'], basic_stats['std'])
        plt.title(title)
        ndev.axvline(basic_stats['median'], color='red', label='Media')
        plt.show()
    
    def plot_normal_distr(self, data_to_test):
        _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        plt.suptitle(f'Normal distribution validation for: {data_to_test.name}')
        
        # histogram with normal
        sns.histplot(data_to_test, kde=True, stat='density', ax=ax1)
        mu, sigma = data_to_test.mean(), data_to_test.std()
        dt_min = data_to_test.min()
        dt_max = data_to_test.max()
        med = data_to_test.median()
        x = np.linspace(dt_min, dt_max, 100)
        ax1.plot(x, sp.stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Perfect Normal Curve')
        ax1.axvline(med, color='green', linestyle='--', linewidth=2, label='Median')
        ax1.axvline(mu, color='orange', linestyle='--', linewidth=2, label='Mean')
        ax1.set_title('Data vs Theoretical Distribution')
        ax1.legend()

        # q-q plot
        stats.probplot(data_to_test, dist="norm", plot=ax2)
        ax2.set_title('Perfect Normal Q-Q Graph')

        # boxplot
        sns.boxplot(y=data_to_test, ax=ax3)
        ax3.set_title('Data Spread and Outliers')

        # kde vs normal
        sns.kdeplot(data_to_test, ax=ax4, label='Sample', clip=(0, None))
        ax4.plot(x, sp.stats.norm.pdf(x, mu, sigma), 'r-', label='Perfect Bell Curve')
        ax4.axvline(med, color='green', linestyle='--', linewidth=2, label='Median')
        ax4.axvline(mu, color='orange', linestyle='--', linewidth=2, label='Mean')
        ax4.set_title('KDE vs Normal')
        ax4.legend()

    def plot_column_distribution(self, column: str, dist_type: str = 'binomial', 
                            **kwargs) -> None:
        if column not in self.da.df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
         
        title = kwargs.get('title', f'{dist_type.title()} Distribution - {column}')
        data = self.da.df
        plt.figure(figsize=(10, 6))
        sns.histplot(data, stat='probability', discrete=True, color='skyblue')
        plt.title(title)
        plt.xlabel('Values')
        plt.ylabel('Probability')
        plt.show()

    def plot_regression_model(self, x: str, y: str) -> None:
        current_data = self.da.df
        sns.regplot(x=x, y=y, data=current_data, scatter_kws={'alpha':0.6}, line_kws={'color': 'red'})