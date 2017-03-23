import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.externals import joblib
import matplotlib.pyplot as plt

# Load data from CSV files
# Test set: as above, except it does not have mortality_rate
features = ['O3','PM10','PM25','NO2','T2M'] # ignore region & date and for now
complete_data = pd.read_csv('./Data/train.csv', parse_dates = ['date'])
test = pd.read_csv('./Data/test.csv')

# Dataset delete missing rows
train = complete_data.dropna(axis=0, how = 'any')

# Get dates
dates = train['date']

# Compute correlation matrix
X_train = train[features]
var_correlation = X_train.corr()

# Train model on PM25
features_PM25 = ['mortality_rate','O3','PM10','T2M']
X_train_PM25 = np.array(train[features_PM25])
Y_train_PM25 = np.array(train['PM25'])

# Regression
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1, verbose=True, cache_size=4000)
rbf = svr_rbf.fit(X_train_PM25, Y_train_PM25)
y_rbf = rbf.predict(X_train_PM25)

mse = ((y_rbf - Y_train_PM25) ** 2).mean(axis=0)
print("MSE is:")
print(mse)
joblib.dump(rbf, './Data/rbf_PM25.pkl') 