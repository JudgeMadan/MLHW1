import numpy as np

# @return X_train, Y_train, X_test, Y_test
def importData(trainName = 'propublicaTrain.csv', testName='propublicaTest.csv'):
    dataTrain = np.genfromtxt(trainName, delimiter=',', skip_header=1, dtype='int64')
    dataTest = np.genfromtxt(testName, delimiter=',', skip_header=1, dtype='int64')

    return (dataTrain[:,1:], dataTrain[:,0], dataTest[:,1:], dataTest[:,0])
