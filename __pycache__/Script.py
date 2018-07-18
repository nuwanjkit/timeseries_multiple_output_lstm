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
import json
import os
import time

def createDir(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def jsonParser(jsn):
    #js='{"name":"nuwan","data":[-15.5,-9.88,-0.5]}'
    data = json.loads(jsn)
    arr=[]

    for element in data['data']:
        arr.append(element)
    #print(arr)
    res=predict(arr,"nuwan")
    return res

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-3):
        a = dataset[i:(i+look_back), 0]
        b = dataset[(i+look_back):(i+look_back+3), 0]
        dataX.append(a)
        dataY.append(b)
    return numpy.array(dataX), numpy.array(dataY)


def saveScaler(scalerName,scaler):
    #scaler_filename = "scaler.save"
    joblib.dump(scaler, scalerName)


def loadScaler(scalerName):
    #scaler = joblib.load(self.scaler_name)
    scaler = joblib.load(scalerName)
    return scaler


def readDatasetAndNormalize(dataSetName,scalerName):
    # code starts from here
    dataframe = read_csv(dataSetName, usecols=[1], engine='python')
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    print(dataset)

    # scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # scaler.fit(dataset)
    saveScaler(scalerName,scaler)
    return dataset

# dataset: input dataset to be trained
# modelname: string name for the model file to be savd


def trainModel(dataSetName,userName):
    try:
        #get the location of files
        my_path = os.path.abspath(os.path.dirname(__file__))
        model_name="./"+userName+"/model.h5"
        scaler_name="./"+userName+"/scaler.sc"
        dataset_name="./"+dataSetName

        modelPath = os.path.join(my_path, model_name)
        scalerPath=os.path.join(my_path,scaler_name)
        datasetPath=os.path.join(my_path,dataset_name)

        #get the dataset by calling "readDatasetAndNormalize" method
        dataset = readDatasetAndNormalize(datasetPath,scalerPath)

        # split into train and test sets
        train_size = int(len(dataset) * 0.67)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size,
                              :], dataset[train_size:len(dataset), :]
        print(len(train), len(test))

        # reshape into X=t and Y=t+1
        look_back = 3
        trainX, trainY = create_dataset(train, look_back)
        print('trainY: ', trainY)
        testX, testY = create_dataset(test, look_back)

        # reshape input to be [samples, time steps, features]
        trainX = numpy.reshape(
            trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        # create and fit the LSTM network
        model = Sequential()
        model.add(LSTM(50, input_shape=(1, look_back)))
        model.add(Dense(3))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, trainY, epochs=5, batch_size=1, verbose=2)

        #save the model in the user's directory
        model.save(modelPath)

    except IOError:
        return "Error occured when reading file!"
    except:
        return "An Error Occured!"



# data: [nu1,nu2,nu3] array
def predict(data,userName):
    try:
        #load the files to prediction
        my_path = os.path.abspath(os.path.dirname(__file__))
        model_name="./"+userName+"/model.h5"
        scaler_name="./"+userName+"/scaler.sc"

        modelPath = os.path.join(my_path, model_name)
        scalerPath=os.path.join(my_path,scaler_name)

        start_time = time.time()
        #load the model
        model = load_model(modelPath)
        scaler = loadScaler(scalerPath)
        print("--- %s seconds ---" % (time.time() - start_time))


        # convert dataset
        # convert to pd.dataframe
        dataframe = pd.DataFrame(data)

        # get values
        values = dataframe.values

        # convert into float
        floatDataset = values.astype('float32')

        # scale the array
        scaledArray = scaler.transform(floatDataset)
        # print(scaledArray)
        nparr = numpy.array(scaledArray)
        #scaled=numpy.reshape(nparr, (nparr.shape[0], 1, nparr.shape[1]))
        # print(nparr.shape[0],nparr.shape[1])
        # change the dim of array

        flatArray = nparr.flatten()
        expandedArray = numpy.expand_dims(flatArray, axis=0)

        output = numpy.reshape(
            expandedArray, (expandedArray.shape[0], 1, expandedArray.shape[1]))
        # make predictions
        testPredict = model.predict(output)
        print("--- %s seconds ---" % (time.time() - start_time))

        # invert predictions
        testPredictInverted = scaler.inverse_transform(testPredict)
        outPandasArray = testPredictInverted[0]
        outArray = numpy.array(outPandasArray)
        outputArray = [outArray[0], outArray[1], outArray[2]]
        # print(outputArray)
        return outputArray

    except IOError:
        return "Error in reading the file!"


trainModel("real_data.csv","nuwan")

#print(predict([-15.5,-9.88,-0.5],"nuwan"))
#createDir('./nuwan/')
