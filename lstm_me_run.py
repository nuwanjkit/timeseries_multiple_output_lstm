from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
import numpy
import math
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv
import matplotlib.pyplot as plt


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
#print(dataset)

# scaler
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
#print(len(train), len(test))


# reshape into X=t and Y=t+1
look_back = 3
trainX, trainY = create_dataset(train, look_back)
#print('trainY: ',trainY)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
model=load_model('new_lstm.h5')

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
testPredict = scaler.inverse_transform(testPredict)
testY=scaler.inverse_transform(testY)
#for i in range(200):
    #print(testY[i],testPredict[i])

test_for_graph=scaler.inverse_transform(test[:200])
a=test_for_graph.reshape(1,200)
#print(test_for_graph)
print(a)
b=a.tolist()
line=b[0]
x=[i for i in range(len(line))]
plt.plot(x,line)

for i in range(198):
    x_i=[(2+i),(3+i),(4+i),(5+i)]
    y_i=[line[2+i]]+ (testPredict[i]).tolist()
    plt.plot(x_i,y_i,color='red')

plt.show()