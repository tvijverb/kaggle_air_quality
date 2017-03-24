import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load data from CSV files
# Test set: as above, except it does not have mortality_rate
features = ['O3','PM10','PM25','NO2','T2M'] # ignore region & date and for now
complete_data = pd.read_csv('./Data/train.csv', parse_dates = ['date'])
test = pd.read_csv('./Data/test.csv')
rbf_PM25 = joblib.load('./Data/rbf_PM25.pkl')
rbf_NO2 = joblib.load('./Data/rbf_NO2.pkl')

# Dataset delete missing rows
complete_data = complete_data.dropna(subset = ['mortality_rate','O3','PM10','T2M'])
complete_data_wo_missings = complete_data.dropna(subset = ['PM25','NO2'])
print("complete_data size: ",complete_data.shape)
print("complete_data w/o missings size: ",complete_data_wo_missings.shape)

# Data with missings
null_data = complete_data[complete_data.isnull().any(axis=1)]
null_data_shape = null_data.shape;
print("null-data size: ",null_data_shape)

# Select rows with PM25 & NO2 missings
features_SVR = ['mortality_rate','O3','PM10','T2M']
X_test = null_data[features_SVR]
print("X_test shape: ", X_test.shape)

# Predict PM25 & NO2 missings
y_hat_PM25 = rbf_PM25.predict(X_test)
y_hat_NO2 = rbf_NO2.predict(X_test)
y_hat_PM25 = y_hat_PM25.reshape((len(y_hat_PM25),1))
y_hat_NO2 = y_hat_NO2.reshape((len(y_hat_NO2),1))
print("B size: ",y_hat_PM25.shape)


# Replace NaN with prediction
append1 = np.hstack((null_data,y_hat_PM25))
append2 = np.hstack((append1,y_hat_NO2))
df = pd.DataFrame(append2)
df = df.drop([6,7],1)
names = df.columns.tolist()
new_names = ['Id','region','date','mortality_rate','O3','PM10','T2M','PM25','NO2']
df.rename(columns=dict(zip(names, new_names)), inplace=True)
df = df[['Id','region','date','mortality_rate','O3','PM10','PM25','NO2','T2M']]
print("new size: ",df.shape)
df.to_csv('./Data/imputed_data.csv', sep=',')


