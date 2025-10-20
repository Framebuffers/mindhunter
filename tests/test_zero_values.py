import pandas as pd
import numpy as np
from faker import Faker
from mindhunter import StatFrame
import pytest
import random


@pytest.fixture
def sample_sf():
    fake = Faker()
    rand = random.Random()
    records = 50
    data = []
    rand_zeros = random.randint(5, records-1)
    for _ in range(records):
        record = {
            'name': fake.name_nonbinary,                # string
            'email': fake.email,                        # string
            'category': fake.boolean(25),               # categoric
            'weight': rand.uniform(0, 200.0),           # numerical
            'height': rand.uniform(0, 220.0),           # numerical
            'age': rand.randint(18, 90),                # numerical (int)
            'zero': 0,                                  # zero-values 
            'nan': np.nan                               # NaN
        }
        data.append(record)
    
    sf = StatFrame(pd.DataFrame(data))
    
    """
        Process:
            Insert zeros at random only on numerical columns.
            The SF will already have a column full of zero values.
            It will assert if:
                - That column is 0
                - All numerical columns have at least one zero value. The DF was adapted to have the chance to have 0 as a valid value, just so it doesn't fail the test just because the rng didn't roll a 0.
    """
     
    numerical_cols = sf.df.select_dtypes(include=[np.number]).columns.to_list()
    rows = np.random.randint(0, len(sf.df), rand_zeros)
    cols_indices = np.random.randint(0, len(numerical_cols), rand_zeros)
    cols = [numerical_cols[i] for i in cols_indices]
    
    for r, c in zip(rows, cols):
        sf.df.at[r, c] = 0
    
    sf.df['nan'] = np.nan
    return sf

def test_locate_zero_rows(sample_sf):
    """
        Test if it can locate columns with zero values.
        
        Process:
            - Creates a SF
            - Checks if:
                - The zero column remains zero
                - All columns have, at least, one zero value.
                - All non-numerical columns don't have a single 0 value.

    """
    assert sample_sf is not None
    
    df = sample_sf.df
    
    assert (df==0).any().any()
    
    zr = sample_sf.locate_zero_rows()
    assert zr is not None
    
def test_analyze_zero_removal(sample_sf):
    assert sample_sf is not None
    results = sample_sf.analyze_zero_removal()
    
    assert results is not None
    for i in sample_sf.df:
            if isinstance(sample_sf.df[i], (np.number, np.integer, np.floating, np.complexfloating)):
                # it should only be numerics that have a 0 on them.
                assert sample_sf.df[i] != 0
    
    assert results.any().any()

def test_remove_exact_zeros(sample_sf):
    records = len(sample_sf.df)/2
    rand_zeros = random.randint(5, int(records-1))
    numerical_cols = sample_sf.df.select_dtypes(include=[np.number]).columns.to_list()
    rows = np.random.randint(0, len(sample_sf.df), rand_zeros)
    cols_indices = np.random.randint(0, len(numerical_cols), rand_zeros)
    cols = [numerical_cols[i] for i in cols_indices]

    for r, c in zip(rows, cols):
        sample_sf.df.at[r, c] = np.float64(random.uniform(0.00001, 0.001))
    
    sample_sf.df['nan'] = np.nan
    
    result = sample_sf.remove_exact_zeros()
    assert result['rows_removed'] is not None
    
    assert (sample_sf.df.select_dtypes(include=np.number) != 0).all().all()

