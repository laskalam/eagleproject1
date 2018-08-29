import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

np.seed = 123456789
def train_validate_split(df, train_percent=.6, seed=None):
    """
    This function randomly divides data into 2 separate sets
    (i.e. train & validation)
    """
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    train = df.ix[perm[:train_end]]
    validate = df.ix[perm[train_end:]]
    
    return train, validate
