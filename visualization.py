import matplotlib.pyplot as plt
from knn import knn
from data_import import importData
import numpy as np
# from mle import mle
# from bayes import bayes


def knnTest(data, kRange=21, kStep=2, debug=False):
    X_train, Y_train, X_test, Y_test = data

    accuracy = np.empty((0,4), dtype='float64') # (k, L1-acc, L2-acc, L3-acc)

    for k in range(1, kRange+1, kStep):
        print("Testing KNN: " + str(k) + "/" + str(kRange))
        predL1 = (knn(X_train, Y_train, X_test, k, 'L1') == Y_test).mean()
        predL2 = (knn(X_train, Y_train, X_test, k, 'L2') == Y_test).mean()
        predLinf = (knn(X_train, Y_train, X_test, k, 'Linf') == Y_test).mean()
        accuracy = np.vstack((accuracy, (k, predL1, predL2, predLinf)))

    return accuracy

def knnVisualize(acc, width=0.2):
    knn_plot = plt.subplot(111)
    knn_plot.bar(acc[:,0]-width, acc[:,1], width=width, color='b', align='center', label="L1 Norm")
    knn_plot.bar(acc[:,0], acc[:,2], width=width, color='g', align='center', label="L2 Norm")
    knn_plot.bar(acc[:,0]+width, acc[:,3], width=width, color='r', align='center', label="L_inf Norm")
    knn_plot.legend()
    knn_plot.set_xlabel('k')
    knn_plot.set_ylabel('Accuracy (%)')
    knn_plot.set_xticks(acc[:,0])
    knn_plot.set_title("Effects of k, L-norm on Accuracy")
    plt.show()






if __name__ == '__main__':
    data = importData()
    knnVisualize(knnTest(data, 37, 2), width=0.42)
