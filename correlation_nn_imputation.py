import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#Load keras stuff
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasRegressor
from keras import losses
from keras import backend as K

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

# define base mode
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(4, input_dim=4, kernel_initializer="uniform", activation="relu"))
	model.add(Dense(1, kernel_initializer="uniform"))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
	
def larger_model():
	# create model
	model = Sequential()
	model.add(Dense(4, input_dim=4, init='normal', activation='relu'))
	model.add(Dense(2, init='normal', activation='relu'))
	model.add(Dense(1, init='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
	
def wider_model():
	# create model
	model = Sequential()
	model.add(Dense(20, input_dim=4, init='normal', activation='relu'))
	model.add(Dense(1, init='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
	
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, nb_epoch=50, batch_size=5, validation_split=0.2, verbose=1)))
pipeline = Pipeline(estimators)

#kfold = KFold(n_splits=5, random_state=seed)
#results = cross_val_score(pipeline, X_train_PM25, Y_train_PM25, cv=kfold)
scores = pipeline.evaluate(X_train_PM25, Y_train_PM25)
print("")
#print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
print("Results: %.2f (%.2f) MSE" % (scores.mean(), scores.std()))

# Model parameters
#model = Sequential()
#model.add(Dense(12, input_dim=4, init='uniform', activation='relu'))
#model.add(Dense(8, init='uniform', activation='relu'))
#model.add(Dense(1, init='uniform', activation='sigmoid'))
# Compile model
#model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# Fit the model
#model.fit(X_train_PM25, Y_train_PM25, nb_epoch=150, batch_size=10, validation_split=0.2)
# evaluate the model
#scores = model.evaluate(X_train_PM25, Y_train_PM25)
#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



K.clear_session()
print(dates.shape)
print(var_correlation.shape)
print(var_correlation)
np.savetxt("corr_matrix.csv", var_correlation, delimiter=",")



