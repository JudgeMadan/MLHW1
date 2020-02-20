import numpy as np

def importData(trainName = 'propublicaTrain.csv', testName='propublicaTest.csv', returnType='normal', removeRedundant=True, ignoreSensitive=False): #0 for split, 1 for same
    dataTrain = np.genfromtxt(trainName, delimiter=',', skip_header=1, dtype='int64')
    dataTest = np.genfromtxt(testName, delimiter=',', skip_header=1, dtype='int64')

    if(removeRedundant):
        dataTrain = dataTrain[:,:-1]
        dataTest = dataTest[:,:-1]

    if(ignoreSensitive):
        dataTrain = np.delete(dataTrain, 3, 1)
        dataTest = np.delete(dataTest, 3, 1)
        
    # if(normalize):
    #     maxbyCol = np.maximum(dataTrain.max(axis=0), dataTest.max(axis=0))
    #     dataTrain = dataTrain / maxbyCol
    #     dataTest = dataTest / maxbyCol

    if(returnType == 'split'): #@return X_train, Y_train, X_test, Y_test
        return (dataTrain[:,1:], dataTrain[:,0], dataTest[:,1:], dataTest[:,0])
    elif(returnType == 'mle'): #@return train (rec_id == 0), train (rec_id==1), X_test, Y_test
        x_0 = dataTrain[np.where(dataTrain[:,0] == 0),1:][0]
        x_1 = dataTrain[np.where(dataTrain[:,0] == 1),1:][0]
        return (x_0, x_1, dataTest[:,1:], dataTest[:,0])
    else: #@return train, test
        return (dataTrain, dataTest)


if __name__ == '__main__':
    print(importData(returnType='mle'))
