from mindhunter.core import DataAnalyzer
import pandas as pd
from faker import Faker
import random
import numpy as np

fake = Faker()

num_records = 10

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
    print(record)

df = pd.DataFrame(data)
assert df is not None

da = DataAnalyzer(df)
assert da is not None

print(da.describe_columns('age'))
print(da._cached_stats)
