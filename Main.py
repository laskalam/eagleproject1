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


data = '../data/train.csv'
train_data = pd.read_csv(data)
data = '../data/test.csv'
test_data = pd.read_csv(data)
#
#TO BE CHANGED
# Handle missing values, feature selection
train_data.isnull().sum()[train_data.isnull().sum()>0]
#Feature_dropped = ['PoolQC','Fence','MiscFeature','Alley','FireplaceQu']
#train_data = train_data.drop(Feature_dropped, axesis=1)
#test_data = test_data.drop(Feature_dropped, axis=1)
model_predictors = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']

Train, Validate = split(train_data, train_percent=TRAINING_PERCENT, seed=SEED)
#
X_train, Y_train = Train[model_predictors], Train.SalePrice
plt.clf()
fig, axes = plt.subplots(3,3, sharex='col', sharey='row')
x_lim = 0,800000
y_lim = 0,800000
k = -1
for i, Model in enumerate(MODELS):
    Regressor, Name, Type = GetRegressor(Model)
    regressor_ = Regressor()
    regressor_.fit(X_train, Y_train)
    
    prediction0 = regressor_.predict(X_train)
    prediction1 = regressor_.predict(Validate[model_predictors])
    
    
    color0 = 'blue'
    color1 = 'lightgreen'
    metric0 = metric(prediction0, Y_train)[0]
    metric1 = metric(prediction1, Validate.SalePrice)[0]
    
    
    
    
    # Plot predictions
    j = i%3
    if j == 0:
        k += 1
    axes[k,j].scatter(Y_train, prediction0, c = color0, marker = "s", label = "Training")
    axes[k,j].scatter(Validate.SalePrice, prediction1, c = color1, marker = "s", label = "Validation")
    axes[k,j].text(0.9, 0.2, '%.f'%metric0, color=color0, ha= 'right', va='bottom', transform=axes[k,j].transAxes)
    axes[k,j].text(0.9, 0.1, '%.f'%metric1, color=color1, ha= 'right', va='bottom', transform=axes[k,j].transAxes)
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


    #file_name = '%s.pdf saved'%Name
    #plt.savefig('%s.pdf'%Name, bbox_inches='tight')
file_name = 'All.pdf saved'
plt.savefig('All.pdf', bbox_inches='tight')
print (file_name, 'saved')
    #plt.show()

#
#Regressor.fit(X_train, Y_train)
#validation = Regressor.predict(Validate)
#
#rmse, Pearsonr = metric()
