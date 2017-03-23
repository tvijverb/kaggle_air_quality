import pandas as pd

# Train set has these columns:
# Id,region,date,mortality_rate,O3,PM10,PM25,NO2,T2M

# Test set: as above, except it does not have mortality_rate
features = ['mortality_rate','O3','PM10','PM25','NO2','T2M'] # ignore region & date and for now

from sklearn.linear_model import LinearRegression

train = pd.read_csv('./train.csv', parse_dates = ['date'])
test = pd.read_csv('./test.csv')

# We have missing values for NO2 and PM25 for 2007.
# Simplest way of dealing with it is to 
# remove the rows with missing values:
train = train.dropna(axis=0, how = 'any')

X_train = train[features]
y = train['mortality_rate'].copy()

#X_test = test[features].copy()

# Correlation stuff here
var_correlation = X_train.corr()

print(var_correlation)
lr = LinearRegression()
lr.fit(X_train, y)

predictions = test[['Id']].copy()
#predictions['mortality_rate'] = lr.predict(X_test)

#predictions.to_csv('linear_regression.csv', index = False)
