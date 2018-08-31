import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
SEED = 123456789
TRAINING_PERCENT = 0.7
Model = 'ABR'
#Choice:
MODELS= 'RF', 'ABR', 'BR', 'GBR', 'OLS', 'SVM', 'HR', 'DTR', 'ETR' #SGD

from train_validate_data import train_validate_split as split
#from FeatureSelection import SelectFeature
from SelectModel import GetRegressor
from metric_score import AccuracyMetric as metric
from ProcessTheData import ProcessTheData
from CONFIG import *

data = '../data/train.csv'
#train_data = pd.read_csv(data)
#print list(train_data)
#sys.exit()
#sys.exit()
train_data, CatNum_features, final_feature  = ProcessTheData(data, miss_threshold=MISS_THRESHOLD,
                                  corr_threshold=CORR_THRESHOLD)
data = '../data/test.csv'
test_data = ProcessTheData(data, miss_threshold=MISS_THRESHOLD, corr_threshold=CORR_THRESHOLD,
                                 final_features=final_feature,catnum_features=CatNum_features)
#
#TO BE CHANGED
#model_predictors = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
Test_ID = test_data.Id
#X_test = test_data[model_predictors]

Train, Validate = split(train_data, train_percent=TRAINING_PERCENT, seed=SEED)

sys.exit()
#Y_train = np.log10(Train['SalePrice'])
Y_train = Train['SalePrice']
X_train = Train.drop(['Id', 'SalePrice'], axis = 1, inplace = True)#[model_predictors]

#Y_validate = np.log10(Validate['SalePrice'])
Y_validate = Validate['SalePrice']
X_validate = Validate.drop(['Id', 'SalePrice'], axis = 1, inplace = True)#[model_predictors]
#Drop NaN in price
#Validate.dropna(axis=0)
#Validate = Validate[~Validate.isin([np.nan, np.inf, -np.inf]).any(1)]
#Train = Train[~Train.isin([np.nan, np.inf, -np.inf]).any(1)]
#X_train = Train[model_predictors]
#Y_train = Train.SalePrice
#X_train.drop( ["Id", "SalePrice"], axis = 1, inplace = True)
#_validate = Validate.copy()
#_validate.drop(["Id", "SalePrice"], axis=1, inplace = True)
#_validate[model_predictors]
#Do some plots
plt.clf()
fig, axes = plt.subplots(3,3, sharex='col', sharey='row')
x_lim = 4,7
y_lim = 4,7
k = -1
for i, Model in enumerate(MODELS):
    Regressor, Name, Type = GetRegressor(Model)
    regressor_ = Regressor()
    regressor_.fit(X_train, Y_train)
    
    prediction0 = regressor_.predict(X_train)
    prediction1 = regressor_.predict(X_validate)
    
    
    color0 = 'blue'
    color1 = 'lightgreen'
    metric0 = metric(prediction0, Y_train)[0]
    metric1 = metric(prediction1, Y_validate)[0]
    Prediction = regressor_.predict(X_test)
    
    
    
    # Plot predictions
    j = i%3
    if j == 0:
        k += 1
    axes[k,j].plot(Y_train, prediction0, '.', color = color0, label = "Training")
    axes[k,j].plot(Y_validate, prediction1, '.', color = color1, label = "Validation")
    axes[k,j].text(0.9, 0.2, '%.4f'%metric0, color=color0, ha= 'right', va='bottom', transform=axes[k,j].transAxes)
    axes[k,j].text(0.9, 0.1, '%.4f'%metric1, color=color1, ha= 'right', va='bottom', transform=axes[k,j].transAxes)
    axes[k,j].text(0.01, 0.9, '%s'%Name, color='k', ha= 'left', va='top', fontsize=8, transform=axes[k,j].transAxes)
    #axes[k,j].set_title(Name)
    if k==2:
        axes[k,j].set_xlabel("Real values")
    if not j:
        axes[k,j].set_ylabel("Predicted values")
    if not i: #axes[k,j].legend(loc = "upper left")
        axes[k,j].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=1, mode="expand", borderaxespad=0.)
    axes[k,j].plot(x_lim, x_lim, c = "red")
    axes[k,j].set_xlim(x_lim)
    axes[k,j].set_ylim(y_lim)

    output = 'output%d.csv'%i
    #print Test_ID
    #print Prediction
    np.savetxt(output, np.c_[Test_ID, 10**Prediction], fmt=['%d', '%f'], delimiter=',', comments='', header='Id,SalePrice')
    print output, 'saved'
    #file_name = '%s.pdf saved'%Name
    #plt.savefig('%s.pdf'%Name, bbox_inches='tight')
file_name = 'All1.pdf'
plt.savefig(file_name, bbox_inches='tight')
print (file_name, 'saved')
    #plt.show()

#
#Regressor.fit(X_train, Y_train)
#validation = Regressor.predict(Validate)
#
#rmse, Pearsonr = metric()
