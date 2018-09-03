from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import numpy as np


def AccuracyMetric(y_actual, y_predicted):
	"""
	** implements a root mean squared error & Correlation Coefficient functions
	 """
	rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))
	PearsonCoeff = np.corrcoef(y_actual,y_predicted)[0,1]
	
	return rmse, PearsonCoeff

def rmseCV(model, train_data, y_train, n_folds=5):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_data.values)
    rmse= np.sqrt(-cross_val_score(model, train_data.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

