# imports
from pandas import read_csv
import numpy
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
import pandas as pd

class ML:
    # convert an array of values into a dataset matrix
    def create_dataset(self,dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-3):
            a = dataset[i:(i+look_back), 0]
            b = dataset[(i+look_back):(i+look_back+3), 0]
            dataX.append(a)
            dataY.append(b)
        return numpy.array(dataX), numpy.array(dataY)

    def saveScaler(self,scaler_name):
        #scaler_filename = "scaler.save"
        self.scaler_name=scaler_name
        joblib.dump(self.scaler, scaler_name)

    def loadScaler(self):
        scaler = joblib.load(self.scaler_name)
        return scaler

    def readDatasetAndNormalize(self):

        # code starts from here
        dataframe = read_csv('real_data.csv', usecols=[1], engine='python')
        dataset = dataframe.values
        dataset = dataset.astype('float32')
        print(dataset)

        # scaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        self.scaler=scaler
        return dataset

    #dataset: input dataset to be trained
    #modelname: string name for the model file to be savd
    def trainModel(self,dataset,modelname):
        # split into train and test sets
        train_size = int(len(dataset) * 0.67)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
        print(len(train), len(test))


        # reshape into X=t and Y=t+1
        look_back = 3
        trainX, trainY = self.create_dataset(train, look_back)
        print('trainY: ', trainY)
        testX, testY = self.create_dataset(test, look_back)

        # reshape input to be [samples, time steps, features]
        trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        # create and fit the LSTM network
        model = Sequential()
        model.add(LSTM(100, input_shape=(1, look_back)))
        model.add(Dense(3))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

        model.save(modelname)

    #testData: numpy array
    def batchPredict(self,testData):
        model=load_model('new_lstm.h5')
        # make predictions
        testPredict = model.predict(testData)

        #load scaler
        scaler=self.loadScaler()

        # invert predictions
        testPredictInverted = scaler.inverse_transform(testPredict)
        print(testPredictInverted)

    #data: [nu1,nu2,nu3] array
    def predict(self,data):
        model=load_model('new_lstm.h5')
        scaler=self.loadScaler()

        #convert dataset
        #convert to pd.dataframe
        dataframe=pd.DataFrame(data)

        #get values
        values=dataframe.values

        #convert into float
        floatDataset=values.astype('float32')

        #scale the array
        scaledArray=scaler.fit_transform(floatDataset)

        # make predictions
        testPredict = model.predict(scaledArray)

        # invert predictions
        testPredictInverted = scaler.inverse_transform(testPredict)
        print(testPredictInverted)

test=ML()
test.predict([-13.5,-15.5,-12.69])