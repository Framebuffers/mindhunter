import StatisticalObject as so
import pandas as pd

class PlottableObject:
    """ `StatisticalObject` with plotting capabilities.
        
        Args:
            `stat`: A StatisticalObject.
        
        Returns:
            A PlottableObject, based on the DataFrame held inside the input StatisticalObject    
    
    """

    def __init__(self, stat: so.StatisticalObject) -> None:
        self.stat_obj = stat

    def get_statisticalobject(self) -> so.StatisticalObject:
        return self.stat_obj
    
    def get_df(self) -> pd.DataFrame:
        return self.stat_obj.statistical_object_df
    
    def update_df(self, new_stat_obj: so.StatisticalObject) -> None:
        self.stat_obj = new_stat_obj

    def get_stats(self, column: str):
        # to avoid having two sources of truth, this delegates the cache of this object directly to the StatisticalObject being wrapped by this PlottableObject.
        return self.stat_obj.get_quick_stats(column)