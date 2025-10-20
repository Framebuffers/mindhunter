import pandas as pd 
import numpy as np
from faker import Faker
from mindhunter import StatFrame, StatModel
import pytest

@pytest.fixture
def sample_df():
    rand = np.random.default_rng()
    records = 50
    data = []
    
    for _ in range(records):
        record = {
            'cat_a': rand.choice(['A', 'B']),
            'num_a': rand.integers(30, 75),
            'num_b': rand.integers(1, 4),
            'cat_b': rand.choice(['Yes', 'No']),
            'num_c': rand.integers(0, 40),
            'num_d': rand.integers(0, 1),
            'num_e': rand.integers(0, 1),
            'num_f': rand.integers(0, 1),
            'num_g': rand.integers(0, 1),
            'num_h': rand.integers(150, 300),
            'num_i': round(rand.uniform(90, 180), 1),
            'num_j': rand.integers(60, 110),
            'num_k': round(rand.uniform(18, 45), 2),
            'num_l': rand.integers(50, 120),
            'num_m': rand.integers(60, 150),
            'target': rand.integers(0, 1)
        }

        data.append(record)
    
    df = pd.DataFrame(data)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    n_zeros = rand.integers(5, records - 1)
    rows = rand.integers(0, len(df), n_zeros)
    cols_idx = rand.integers(0, len(numeric_cols), n_zeros)
    
    for r, c_idx in zip(rows, cols_idx):
        df.at[r, numeric_cols[c_idx]] = 0
    
    return StatFrame(df)

# failing
def test_lineal_regression(sample_df: StatFrame):
    sm = StatModel(sample_df)
    
    sm.train_linear_model(
        'cat_b'
    )