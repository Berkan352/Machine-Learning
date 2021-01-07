import keras
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import keras
from keras.layers import Dense,Flatten,Conv3D,Conv2D,Activation,MaxPooling2D,Dropout
from keras import Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
warnings.filterwarnings("ignore")

(X_train,y_train),(X_test,y_test)=keras.datasets.fashion_mnist.load_data()#loading fashion_mnist dataset

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)#resize images to keras need
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train/=255#normalization
X_test/=255

X_train.shape

Y_train = np_utils.to_categorical(y_train, 10)# one-hot encoding
Y_test = np_utils.to_categorical(y_test, 10)# one-hot encoding

#creating a model
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(28,28,1)))
model.add(Activation('relu'))
BatchNormalization(axis=-1)
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

BatchNormalization(axis=-1)
model.add(Conv2D(64,(3, 3)))
model.add(Activation('relu'))
BatchNormalization(axis=-1)
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())# Fully connected layer


BatchNormalization()
model.add(Dense(512))
model.add(Activation('relu'))
BatchNormalization()
model.add(Dropout(0.2))
model.add(Dense(10))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.fit(X_train,Y_train,steps_per_epoch=48000//64, epochs=10,validation_split=0.2, validation_steps=10000//64)#training

predictions = model.predict_classes(X_test)#getting predictions

predictions = list(predictions)
actuals = list(y_test)

sub = pd.DataFrame({'Actual': actuals, 'Predictions': predictions})
sub.to_csv('./output_cnn.csv', index=False)


