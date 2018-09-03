TYPE = 'RF', 'ABR', 'BR', 'GBR', 'OLS', 'SGD', 'HR', 'DTR', 'ETR'


import sys
import numpy as np
import pandas as pd

def GetRegressor(TYPE):
    '''
    Use:
    GetRegressor(Type)
    #Type should be one of the list above
    Return:
    regressor, regressorname, type of regressor
    '''
    if TYPE in ['RF', 'ABR', 'BR', 'GBR']:
        TypeOfRegression = 'ensemble'
    if TYPE in ['OLS', 'SGD', 'HR']:
        TypeOfRegression = 'linear_model'
    if TYPE in ['DTR', 'ETR']:
        TypeOfRegression = 'tree'
    if TYPE in ['SVM']:
        TypeOfRegression = 'svm'
    #Ensemble
    if TYPE == 'ABR':
        regressor = 'AdaBoostRegressor' 
        from sklearn.ensemble import AdaBoostRegressor as Regressor
    if TYPE == 'BR':
        regressor = 'BaggingRegressor'
        from sklearn.ensemble import BaggingRegressor as Regressor
    if TYPE == 'ETR':
        regressor = 'ExtraTreesRegressor'
        from sklearn.ensemble import ExtraTreesRegressor as Regressor
    if TYPE == 'GBR':
        regressor = 'GradientBoostingRegressor'
        from sklearn.ensemble import GradientBoostingRegressor as Regressor
    if TYPE == 'RF':
        regressor = 'RandomForestRegressor'
        from sklearn.ensemble import RandomForestRegressor as Regressor
    #Linear Model
    if TYPE == 'OLS':
        regressor = 'LinearRegression' #Basic least square
        from sklearn.linear_model import LinearRegression as Regressor
    if TYPE == 'HR':
        regressor = 'HuberRegressor' #Good for outliers
        from sklearn.linear_model import HuberRegressor as Regressor
    if TYPE == 'SGD':
        regressor = 'SGDRegressor' #Good for outliers
        from sklearn.linear_model import SGDRegressor as Regressor
    #Tree algorithm
    if TYPE == 'DTR':
        regressor = 'DecisionTreeRegressor'
        from sklearn.tree import DecisionTreeRegressor as Regressor
    if TYPE == 'ETR':
        regressor = 'ExtraTreeRegressor'#Extremely randomized tree
        from sklearn.tree import ExtraTreeRegressor as Regressor
    #SVM
    if TYPE == 'SVM':
        regressor = 'LinearSVR'
        from sklearn.svm import LinearSVR as Regressor
    return Regressor, regressor , TypeOfRegression

if __name__ == "__main__":
    for t in TYPE:
        print GetRegressor(t)
