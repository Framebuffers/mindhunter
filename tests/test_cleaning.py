import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp
from faker import Faker
from scipy.stats import norm
from scipy import stats
from mindhunter import StatFrame
import pytest
import random

@pytest.fixture
def sample_statframe():
    fake = Faker()
    rand = random.Random()
    records = 50
    data = []
    for _ in range(records):
        record = {
            'name': fake.name_nonbinary,                # string
            'email': fake.email,                         # string
            'category': fake.boolean(25),               # categoric
            'weight': rand.uniform(30.0, 200.0),        # numerical
            'height': rand.uniform(100.0, 220.0),       # numerical
            'age': rand.randint(18, 90),                # numerical (int)
        }
        data.append(record)
    return StatFrame(pd.DataFrame(data))

def test_clean_sf(sample_statframe: StatFrame):
    
    assert sample_statframe._cached_stats is not None
    """ Check if the StatFrame has been loaded correctly and the internal cache is populated."""
    
    sample_statframe.clean_df()
    assert sample_statframe.df is not None
    """ Check if the DF is loaded, and it can be editable. """