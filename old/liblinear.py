import pandas as pd
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# Load data from CSV files
# Test set: as above, except it does not have mortality_rate
features = ['O3','PM10','PM25','NO2','T2M'] # ignore region & date and for now
complete_data = pd.read_csv('./train.csv', parse_dates = ['date'])
test = pd.read_csv('./test.csv')

# Dataset delete missing rows
train = complete_data.dropna(axis=0, how = 'any')

# Get dates
dates = train['date']

# Compute correlation matrix
X_train = train[features]
var_correlation = X_train.corr()

# Train model on PM25
features_PM25 = ['O3','PM10','NO2','T2M']
X_train_PM25 = np.array(train[features_PM25])
Y_train_PM25 = np.array(train['PM25'])

