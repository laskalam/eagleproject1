import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
SEED = 123456789
TRAINING_PERCENT = 0.7
Model = 'RF'
#Choice:
# 'RF', 'ABR', 'BR', 'GBR', 'OLS', 'SGD', 'HR', 'DTR', 'ETR'

from train_validate_data import train_validate_split as split
from FeatureSelection import SelectFeature
from SelectModel import GetRegressor


data = '../data/train.csv'
train_data = pd.read_csv(data)
data = '../data/test.csv'
test_data = pd.read_csv(data)

Train, Validate = split(train_data, train_percent=TRAINING_PERCENT, seed=SEED)

Regressor, Name, Type = GetRegressor(Model)

Regressor.fit(X_train, Y_train)
validation = Regressor.predict(Validate)
