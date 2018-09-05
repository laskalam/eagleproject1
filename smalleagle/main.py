import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from scipy.special import boxcox1p


from sklearn.model_selection import GridSearchCV
from sklearn import decomposition
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
#from ProcessTheData import ProcessTheData
from metric_score import AccuracyMetric, rmseCV
from ModelRamblings import AveragingModels, StackingModels
from subset import subset_sum

SEED = 1006163

train_path = '../../data/train.csv'
test_path = '../../data/test.csv'
    


newdirectory = 'myTest3'

if not os.path.exists(newdirectory):
    #Create a new directory
    os.makedirs(newdirectory)




def model(train,test, y_train, train_ID, test_ID, fractions = [[0.60, 0.20, 0.20]], file_ID=0): 


    ##RandomForest
    #n = 1000#[20, 100, 200, 500, 1000, 1500, 2000, 2500]
    #RF = make_pipeline(RobustScaler(), RandomForestRegressor(n_estimators=n))
    #
    ##rmse = rmseCV(lasso, pca.transform(train), y_train)
    #rmse = rmseCV(RF, train, y_train)
    #sys.stdout.write("RF rmse(cv): %.4f (std=%.4f)\n"%(rmse.mean(), rmse.std()))
    #
    ## LASSO Regression
    ##parameters = {
    ##             'alpha': np.arange(0.0001, 0.001, 0.0001),
    ##             'random_state':[SEED]
    ##             }
    ##clf = GridSearchCV(Lasso(), parameters)
    ##clf.fit(train,y_train)
    ##print clf.best_params_
    ##sys.exit()
    ##{'alpha': 0.00040000000000000002, 'random_state': SEED}

    #lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0004, random_state=1006163))
    #
    #rmse = rmseCV(lasso, train, y_train)
    #sys.stdout.write("Lasso rmse(cv): %.4f (std=%.4f)\n"%(rmse.mean(), rmse.std()))
    
    ### Elastic Net Regression
    ##parameters = {
    ##             #'alpha': np.arange(0.0001, 0.001, 0.0001),
    ##             'alpha': np.arange(0.0003, 0.0005, 0.00001),
    ##             #'l1_ratio': np.arange(0.1, 1, 0.1),
    ##             'l1_ratio': np.arange(0.8, 1, 0.01),
    ##             'random_state': [SEED],
    ##             }
    ##clf = GridSearchCV(ElasticNet(), parameters, n_jobs=16)
    ##clf.fit(train,y_train)
    ##print clf.best_params_
    ##sys.exit()
    ##{'alpha': 0.0030000000000000001, 'random_state': 1006163, 'l1_ratio': 0.80000000000000004}
    ##{'alpha': 0.0003500000000000001, 'random_state': 1006163, 'l1_ratio': 0.99000000000000021}
    #ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.00035, l1_ratio=0.99, random_state=1006163))
    #
    #
    #
    #rmse = rmseCV(ENet, train, y_train)
    #sys.stdout.write("ElasticNet rmse(cv): %.4f (std=%.4f)\n"%(rmse.mean(), rmse.std()))
    
    ## Kernel Ridge Regression
    ##parameters = {
    ##            'alpha': np.arange(0.3, 1, 0.1),
    ##            'kernel': ['polynomial'],
    ##            'degree': range(1,6),
    ##            'coef0': np.arange(1,5,0.5)
    ##            }
    ##clf = GridSearchCV(KernelRidge(), parameters, n_jobs=16)
    ##clf.fit(train,y_train)
    ##print clf.best_params_
    ##sys.exit()
    ##{'alpha': 0.80000000000000027, 'coef0': 4.5, 'degree': 2, 'kernel': 'polynomial'}
    #KRR = KernelRidge(alpha=0.8, kernel='polynomial', degree=2, coef0=4.5)
    #
    #rmse = rmseCV(KRR, train, y_train)
    #sys.stdout.write("Kernel Ridge rmse(cv): %.4f (std=%.4f)\n"%(rmse.mean(), rmse.std()))
    
    # Gradient Boosting Regression
    #parameters = {
    #              'n_estimators':range(1000,3500, 500), 'learning_rate':np.arange(0.02,0.07, 0.01),
    #              'max_depth':[3,4,5,6], 'max_features':['sqrt'],
    #              'min_samples_leaf':[5,10,15,20], 'min_samples_split':[5,10,15],
    #              'loss':['huber'], 'random_state':[SEED]
    #              }
    #clf = GridSearchCV(GradientBoostingRegressor(), parameters, n_jobs=16)
    #clf.fit(train,y_train)
    #print clf.best_params_
    #sys.exit()
    #{'loss': 'huber', 'learning_rate': 0.02, 'min_samples_leaf': 5, 'n_estimators': 2500, 
    #'min_samples_split': 15, 'random_state': 1006163, 'max_features': 'sqrt', 'max_depth'    : 3}
    #GBoost = GradientBoostingRegressor(
    #                                   n_estimators=3000, learning_rate=0.05,
    #                                   max_depth=4, max_features='sqrt',
    #                                   min_samples_leaf=15, min_samples_split=10,
    #                                   loss='huber', random_state =5)
    #
    #rmse = rmseCV(GBoost, train, y_train)
    #sys.stdout.write("Gradient Boosting rmse(cv): %.4f (std=%.4f)\n"%(rmse.mean(), rmse.std()))


    ### XGBoost
    ##parameters = {
    ##              'nthread':[2], #when use hyperthread, xgboost may become slower
    ##              'objective':['reg:linear'],
    ##              'learning_rate': np.arange(0.02,0.08, 0.01), #so called `eta` value
    ##              'max_depth': range(2,6),
    ##              'min_child_weight': np.append(np.arange(1,3,0.5),1.7817),
    ##              'colsample_bytree': np.arange(0.2, 0.7,0.1),
    ##              'n_estimators': range(1000, 3000, 500),
    ##              'n_jobs': [4]
    ##             }
    ##clf = GridSearchCV(xgb.XGBRegressor(), parameters, n_jobs=32)
    ##clf.fit(train,y_train)
    ##print clf.best_params_
    ##sys.exit()
    #    {'n_jobs': 4, 'colsample_bytree': 0.20000000000000001, 'learning_rate': 0.039999999999999994, 'nthread': 2, 'min_child_weight': 2.5, 'n_estimators': 1000,     'objective': 'reg:linear', 'max_depth': 3}
    #model_xgb = xgb.XGBRegressor(
    #                            colsample_bytree=0.4603, gamma=0.0468,
    #                            learning_rate=0.05, max_depth=3,
    #                            min_child_weight=1.7817, n_estimators=2200,
    #                            reg_alpha=0.4640, reg_lambda=0.8571,
    #                            subsample=0.5213, silent=1
    #                            #------------
    #                            #colsample_bytree=0.4,
    #                            #gamma=0.045,
    #                            #learning_rate=0.07,
    #                            #max_depth=20,
    #                            #min_child_weight=1.5,
    #                            #n_estimators=300,
    #                            #reg_alpha=0.65,
    #                            #reg_lambda=0.45,
    #                            #subsample=0.95
    #                            )
    #
    #rmse = rmseCV(model_xgb, train, y_train)
    #sys.stdout.write("Xgboost rmse(cv): %.4f (std=%.4f)\n"%(rmse.mean(), rmse.std()))

    ## LightGBM
    #parameters = {
    #             'objective':['regression'],
    #             'num_leaves':[10,20,30],
    #             'learning_rate':[0.05, 0.1, 0.15],
    #             'n_estimators':[200, 400, 600, 700],
    #             'n_jobs': [4],
    #             'subsample_freq': [2,3,4,5],
    #             'reg_alpha': [0.01, 0.02, 0.05],
    #             'reg_lambda': [0.01, 0.02, 0.05],
    #             'random_state':[SEED],
    #             'n_jobs':[4]
    #             #'importance_type': ['split','gain']
    #             }
    #clf = GridSearchCV(lgb.LGBMRegressor(), parameters, n_jobs=16)
    #clf.fit(train,y_train)
    #print clf.best_params_
    #sys.exit()
    #{'num_leaves': 10, 'reg_alpha': 0.01, 'n_jobs': 4, 'subsample_freq': 2, 'learning_rate': 0.05, 'n_estimators': 400, 'reg_lambda': 0.01, 'random_state':        1006163, 'objective': 'regression'}
    model_lgb = lgb.LGBMRegressor(
                                 objective='regression',num_leaves=5,
                                 learning_rate=0.05, n_estimators=720,
                                 max_bin = 55, bagging_fraction = 0.8,
                                 bagging_freq = 5, feature_fraction = 0.2319,
                                 feature_fraction_seed=9, bagging_seed=9,
                                 min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
    
    
    rmse = rmseCV(model_lgb, train, y_train)
    sys.stdout.write("LGBM rmse(cv): %.4f (std=%.4f)\n"%(rmse.mean(), rmse.std()))
    sys.exit()
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
train,test, y_train, train_ID, test_ID = ProcessTheData(train_path, test_path, TopMissingData=10, corr_threshold=None)
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
#model(train,test, y_train, train_ID, test_ID, fractions=ss, file_ID=M) 
model(train,test, y_train, train_ID, test_ID, file_ID=M) 

sys.stdout.write('=====================================\n')
sys.stdout.write('Success EAGLE\n')
sys.stdout.write('=====================================\n')

