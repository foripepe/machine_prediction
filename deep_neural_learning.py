
# coding: utf-8

# In[1]:

# LSTM for sales problem with regression framing

import numpy
import matplotlib.pyplot as plt
import pandas
import math
import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# convert an array of values into a dataset matrix
# def create_dataset(dataset, look_back=1):
#     dataX, dataY = [], []
#     for i in range(len(dataset)-look_back-1):
#         a = dataset[i:(i+look_back), :]
#         dataX.append(a)
#         yz = dataset[i + look_back, 5]
#         dataY.append(yz)
#         print(yz)
#     return numpy.array(dataX), numpy.array(dataY)

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = []
        for j in range(look_back):
            a = numpy.concatenate([a, dataset[i + j][:-1]])
        print(a)
        dataX.append(a)
        dataY.append(dataset[i + look_back, 5])
    return numpy.array(dataX), numpy.array(dataY)

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
dataframe = pandas.read_csv('data/NL Pb till Dec 2016.csv', usecols=[2,3,4,5,6,7], engine='python', skipfooter=2)


# In[3]:

dataset = dataframe.values
dataset = dataset.astype('float32')


# In[5]:

# normalize the dataset
#scaler = MinMaxScaler(feature_range=(0, 1))
#dataset = scaler.fit_transform(dataset)


# In[6]:

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]


# In[ ]:

# reshape into X=t and Y=t+1
look_back = 5
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


# In[174]:

# reshape input to be [samples, time steps, features]
#trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[2], look_back))
#trainX = numpy.reshape(trainX, (trainX.shape[0] * trainX.shape[1], look_back))

#testX = numpy.reshape(testX, (testX.shape[0] * testX.shape[1], look_back))


# In[8]:

# create and fit the LSTM network
model = Sequential()
model.add(Dense(40, input_dim=look_back*5, activation='relu'))
model.add(Dense(40,  activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=100, batch_size=5, verbose=2)


# In[7]:

import sys
sys.setrecursionlimit(10000)
# Save the model
# pickle.dump( model, open( "save.p", "wb" ) )


# In[9]:

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


# In[10]:

testScore = math.sqrt(mean_squared_error(testY, testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

