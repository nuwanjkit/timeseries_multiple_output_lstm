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
    def __init__(self,user_id):
        self.userID=user_id
        self.scalerName=user_id+".sc"
        self.modelName=user_id+".h5"

    # convert an array of values into a dataset matrix
    def create_dataset(self,dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-3):
            a = dataset[i:(i+look_back), 0]
            b = dataset[(i+look_back):(i+look_back+3), 0]
            dataX.append(a)
            dataY.append(b)
        return numpy.array(dataX), numpy.array(dataY)

    def saveScaler(self):
        #scaler_filename = "scaler.save"
        joblib.dump(self.scaler, self.scalerName)

    def loadScaler(self):
        #scaler = joblib.load(self.scaler_name)
        scaler = joblib.load(self.scalerName)
        return scaler

    def readDatasetAndNormalize(self,dataSetName):
        # code starts from here
        dataframe = read_csv(dataSetName, usecols=[1], engine='python')
        dataset = dataframe.values
        dataset = dataset.astype('float32')
        print(dataset)

        # scaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)
        #scaler.fit(dataset)
        self.scaler=scaler
        self.saveScaler()
        return dataset

    #dataset: input dataset to be trained
    #modelname: string name for the model file to be savd
    def trainModel(self,dataSetName):
        try:
            dataset=self.readDatasetAndNormalize(dataSetName)
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
            model.add(LSTM(50, input_shape=(1, look_back)))
            model.add(Dense(3))
            model.compile(loss='mean_squared_error', optimizer='adam')
            model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

            model.save(self.modelName)
        except IOError:
            return "Error occured when reading file!"
        except:
            return "An Error Occured!"

    #data: [nu1,nu2,nu3] array
    def predict(self,data):
        try:
            model=load_model(self.modelName)
            scaler=self.loadScaler()

            #convert dataset
            #convert to pd.dataframe
            dataframe=pd.DataFrame(data)

            #get values
            values=dataframe.values

            #convert into float
            floatDataset=values.astype('float32')

            #scale the array
            scaledArray=scaler.transform(floatDataset)
            print(scaledArray)
            nparr=numpy.array(scaledArray)
            #scaled=numpy.reshape(nparr, (nparr.shape[0], 1, nparr.shape[1]))
            #print(nparr.shape[0],nparr.shape[1])
            #change the dim of array

            flatArray = nparr.flatten()
            expandedArray = numpy.expand_dims(flatArray, axis=0)

            output = numpy.reshape(expandedArray, (expandedArray.shape[0], 1, expandedArray.shape[1]))
            # make predictions
            testPredict = model.predict(output)

            # invert predictions
            testPredictInverted = scaler.inverse_transform(testPredict)
            print(testPredictInverted)

        except IOError:
            return "Error in reading the file!"

test=ML("nuwan")
#dataset=test.readDatasetAndNormalize("real_data.csv")
#test.trainModel(dataset)

test.predict([-15.5,-9.88,-0.5])