from mindhunter.core import DataAnalyzer
from faker import Faker

import os
import pytest
import pandas as pd
import numpy as np

def get_test_size():
    """
    
    Returns dataset size based on environment.
    
    """
    size = os.getenv('TEST_SIZE', 'large')
    return {
        'small': 1000,
        'medium': 10000,
        'large': 100000
    }[size]

@pytest.fixture
def sample_analyzer():
    """
    
    Generate randomized DataFrame based on test environment with both numerical and categorical values.
    
    """
    fake = Faker()
    size = get_test_size()

    test_df =  pd.DataFrame({
        'id': range(size),
        'value': fake.random_elements(elements=range(100), length=size),
        'category': fake.random_elements(elements=['A', 'B', 'C'], length=size)
    })
    
    return DataAnalyzer(test_df)


def test_cache_not_none(sample_analyzer):
    """
    
    Always runs - uses environment-based size.
    
    """
    assert sample_analyzer._cached_stats is not None

@pytest.mark.large
def test_heavy_computation(sample_analyzer: DataAnalyzer):
    """
    
    Skipped in CI - only runs locally.
    Checks running operations directy on the cached DF.
    
    """    
    fake = Faker()

    num_records = 100

    data = []

    for _ in range(num_records):
        record = {
            'first_name': fake.first_name(),
            'last_name': fake.last_name(),
            'email': fake.email(),
            'address': fake.address(),
            'job': fake.job(),
            'age': np.random.randint(18, 99)
        }
        data.append(record)

    df = pd.DataFrame(data)
    assert df is not None

    da = DataAnalyzer(df)
    assert da is not None
 
    result = sample_analyzer._cached_stats
    assert result is not None