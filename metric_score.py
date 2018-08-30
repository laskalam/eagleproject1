from sklearn.metrics import mean_squared_error
import numpy as np


def AccuracyMetric(y_actual, y_predicted):
	"""
	** implements a root mean squared error & Correlation Coefficient functions
	 """
	rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))
	PearsonCoeff = np.corrcoef(y_actual,y_predicted)[0,1]
	
	return rmse, PearsonCoeff
