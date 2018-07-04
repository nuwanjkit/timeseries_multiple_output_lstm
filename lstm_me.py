# imports
from pandas import read_csv
import numpy
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-3):
        a = dataset[i:(i+look_back), 0]
        b = dataset[(i+look_back):(i+look_back+3), 0]
        dataX.append(a)
        dataY.append(b)
    return numpy.array(dataX), numpy.array(dataY)


# code starts from here
dataframe = read_csv('real_data.csv', usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')
print(dataset)

# scaler
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print(len(train), len(test))


# reshape into X=t and Y=t+1
look_back = 3
trainX, trainY = create_dataset(train, look_back)
print('trainY: ',trainY)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(100, input_shape=(1, look_back)))
model.add(Dense(3))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

model.save('new_lstm.h5')

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
testPredict = scaler.inverse_transform(testPredict)

print(testPredict)

