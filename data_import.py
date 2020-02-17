import numpy as np

def importTraining():
    return np.genfromtxt('./propublicaTrain.csv', delimiter=',', skip_header=1)

def importTesting():
    return np.genfromtxt('./propublicaTest.csv', delimiter=',', skip_header=1)

if __name__ == '__main__': # testing code
    print(importTraining())
