import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from scipy.special import boxcox1p




from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

from dataprocessing import ProcessTheData
from metric_score import AccuracyMetric, rmseCV
from ModelRamblings import AveragingModels, StackingModels
from subset import subset_sum

train_path = '../../data/train.csv'
test_path = '../../data/test.csv'
    


newdirectory = 'Test1'

if not os.path.exists(newdirectory):
    #Create a new directory
    os.makedirs(newdirectory)




def model(train,test, y_train, train_ID, test_ID, fractions = [[0.60, 0.20, 0.20]], file_ID=0): 


    # LASSO Regression
    lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
    
    rmse = rmseCV(lasso, train, y_train)
    sys.stdout.write("Lasso rmse(cv): %.4f (std=%.4f)\n"%(rmse.mean(), rmse.std()))
    
    # Elastic Net Regression
    ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
    
    
    
    rmse = rmseCV(ENet, train, y_train)
    sys.stdout.write("ElasticNet rmse(cv): %.4f (std=%.4f)\n"%(rmse.mean(), rmse.std()))
    
    # Kernel Ridge Regression
    KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
    
    rmse = rmseCV(KRR, train, y_train)
    sys.stdout.write("Kernel Ridge rmse(cv): %.4f (std=%.4f)\n"%(rmse.mean(), rmse.std()))
    
    # Gradient Boosting Regression
    GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                       max_depth=4, max_features='sqrt',
                                       min_samples_leaf=15, min_samples_split=10,
                                       loss='huber', random_state =5)
    
    rmse = rmseCV(GBoost, train, y_train)
    sys.stdout.write("Gradient Boosting rmse(cv): %.4f (std=%.4f)\n"%(rmse.mean(), rmse.std()))


    # XGBoost
    model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
    learning_rate=0.05, max_depth=3,
    min_child_weight=1.7817, n_estimators=2200,
    reg_alpha=0.4640, reg_lambda=0.8571,
    subsample=0.5213, silent=1)
    
    rmse = rmseCV(model_xgb, train, y_train)
    sys.stdout.write("Xgboost rmse(cv): %.4f (std=%.4f)\n"%(rmse.mean(), rmse.std()))

    # LightGBM
    model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
    learning_rate=0.05, n_estimators=720,
    max_bin = 55, bagging_fraction = 0.8,
    bagging_freq = 5, feature_fraction = 0.2319,
    feature_fraction_seed=9, bagging_seed=9,
    min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
    
    
    rmse = rmseCV(model_lgb, train, y_train)
    sys.stdout.write("LGBM rmse(cv): %.4f (std=%.4f)\n"%(rmse.mean(), rmse.std()))

    # averaged model
    averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))
    
    rmse = rmseCV(averaged_models, train, y_train)
    sys.stdout.write("Averaged models rmse(cv): %.4f (std=%.4f)\n"%(rmse.mean(), rmse.std()))
    
    # stacked averaged model
    stacked_averaged_models = StackingModels(base_models = (ENet, GBoost, KRR), meta_model = lasso)
    
    rmse = rmseCV(stacked_averaged_models, train, y_train)
    sys.stdout.write("Stacked  models rmse(cv): %.4f (std=%.4f)\n"%(rmse.mean(), rmse.std()))



    # StackedRegressor
    stacked_averaged_models.fit(train.values, y_train)
    stacked_train_pred = stacked_averaged_models.predict(train.values)
    stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
    sys.stdout.write('rmse stacked: %.4f\n'%AccuracyMetric(y_train, stacked_train_pred)[0])
    
    # XGBoost
    model_xgb.fit(train, y_train)
    xgb_train_pred = model_xgb.predict(train)
    xgb_pred = np.expm1(model_xgb.predict(test))
    sys.stdout.write('rmse xgboost: %.4f\n'%AccuracyMetric(y_train, xgb_train_pred)[0])
    
    
    
    # LightGBM
    model_lgb.fit(train, y_train)
    lgb_train_pred = model_lgb.predict(train)
    lgb_pred = np.expm1(model_lgb.predict(test.values))
    sys.stdout.write('rmse LGBM: %.4f\n'%AccuracyMetric(y_train, lgb_train_pred)[0])
    
    #sys.stdout.write('rmse Ensemble (final): %.4f\n'%AccuracyMetric(y_train,stacked_train_pred*0.70 + xgb_train_pred*0.15 + lgb_train_pred*0.15 )[0])
    file_rmse = '%s/rmse.txt'%newdirectory
    file_ = open(file_rmse, 'w')
    for j, frac in enumerate(fractions):
        X, Y, Z = frac
        rmse = AccuracyMetric(y_train,stacked_train_pred*X + xgb_train_pred*Y + lgb_train_pred*Z)[0]
        sys.stdout.write('rmse Ensemble (final): %.4f\n'%rmse)
        file_.write('%.5f\n'%rmse)
        #Final prediction 
        #ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15
        ensemble = stacked_pred*X + xgb_pred*Y + lgb_pred*Z

        sub = pd.DataFrame()
        sub['Id'] = test_ID
        sub['SalePrice'] = ensemble
        output_name = '%s/output%d.csv'%(newdirectory, file_ID+j)
        sub.to_csv(output_name,index=False)
        sys.stdout.write('%s saved \n'%output_name)
    file_.close()
    sys.stdout.write('%s saved \n'%file_rmse)
### MAIN
train,test, y_train, train_ID, test_ID = ProcessTheData(train_path, test_path)
##one can do a feature selection here
#selected_features = ...
#train = train[selected_features]
#test = test[selected_features]
N = 20
fraction_file = '%s/Fractions.txt'%newdirectory
file_ = open(fraction_file, 'w')
#Mika
M = 0
ss = subset_sum(np.arange(0, 1.1, 0.1), 1., init=0)
##Munira
#M = 20
#ss = subset_sum(np.arange(0, 1.1, 0.1), 1., init=20)
##Theo
#M = 40
#ss = subset_sum(np.arange(0, 1.1, 0.1), 1., init=40)
    
#code your proportion of stacked, xgb and lgb
#print '%.2f   %.2f   %.2f\n'%(fractions[0], fractions[1], fractions[2])
for fractions in ss:
    file_.write('%.2f   %.2f   %.2f\n'%(fractions[0], fractions[1], fractions[2]))
file_.close()
sys.stdout.write('%s saved\n'%fraction_file)
model(train,test, y_train, train_ID, test_ID, fractions=ss, file_ID=M) 

sys.stdout.write('=====================================\n')
sys.stdout.write('Success EAGLE\n')
sys.stdout.write('=====================================\n')

