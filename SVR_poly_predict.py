import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Calculate mean mortality rate and use it as the predicted value. 
features = ['O3','PM10','PM25','NO2','T2M'] # ignore region & date and for now
rbf_SVR = joblib.load('./Data/rbf_SVR.pkl')
test = pd.read_csv('./Data/test.csv')

# PreProcess data
X_test = test[features]

# SVR regression - predict Y
predictions = test[['Id']].copy()
predictions['mortality_rate'] = rbf_SVR.predict(X_test)

predictions.to_csv('./Data/svr.csv', index = False)
