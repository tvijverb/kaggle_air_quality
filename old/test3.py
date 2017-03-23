from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

import numpy as np
import h5py

input_num_units = 125000
hidden_num_units = 200
output_num_units = 2

# dataset operations
print('-'*50)
print('Loading data')
file_name = 'hdf5_AML6_H.h5'
file = h5py.File(file_name, 'r')
dataset = file['/data'][()].astype(np.float32)
labels = file['/label'][()]

print('Dataset shape', dataset.shape)
print('Labels shape', labels.shape)

print('Converting labels')
y_train = np_utils.to_categorical(labels)

# create model
print('-'*50)
print('Creating model')
model = Sequential()
model.add(Flatten(input_shape=(50,50,50, 1)))
model.add(Dense(hidden_num_units, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

epochs = 15
lrate = 0.001
decay = lrate/epochs
#sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
adm = Adam(lr=lrate)
model.compile(loss='binary_crossentropy', optimizer=adm, metrics=['accuracy'])
#print(model.summary())

chkpoint = ModelCheckpoint('conv3d_best.h5', monitor='val_acc')
# Fit the model
print('Starting training')
model.fit(dataset, y_train, epochs=epochs, validation_split=0.2, batch_size=1, callbacks=[chkpoint])
# Final evaluation of the model
scores = model.evaluate(dataset, y_train, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
K.clear_session()

