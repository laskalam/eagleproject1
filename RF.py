"""HOUSE PRICE PREDICTION: RANDOM FOREST"""
"""TRAINING THE MODEL"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Importing the dataset
path_train = "../input/train.csv"
data_train = pd.read_csv(path_train)

# Target Matrix
y_train = data_train.SalePrice

# Feature Matrix
model_predictors = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
X_train = data_train[model_predictors]


#Training thr model
regressor_forest = RandomForestRegressor(n_estimators = 70, random_state = 0, min_samples_leaf = 3,
                                    max_features = 'auto')
regressor_forest.fit(X_train, y_train)



"""HOUSE PRICE PREDICTION: RANDOM FOREST"""
"""TESTING OF MODEL"""

#Import dataset
path_test = '../input/test.csv'
data_test = pd.read_csv(path_test)

#Feature Matrix
model_predictors = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
X_test = data_test[model_predictors]

#Prediction
y_pred = regressor_forest.predict(X_test)
print (y_pred)
