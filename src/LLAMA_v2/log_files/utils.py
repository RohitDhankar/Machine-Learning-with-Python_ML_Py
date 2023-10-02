

import pandas as pd
df = pd.read_parquet('train-00000-of-00001-a09b74b3ef9c3b56.parquet')
#train-00000-of-00001-a09b74b3ef9c3b56
df.to_csv('train.csv', index=False)