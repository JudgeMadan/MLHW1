import matplotlib.pyplot as plt
from data_import import importData
from data_process import importData as importData6
import numpy as np

import mle
from bayes import naive_bayes
from knn import knn
from tree import makeTree, testTree


# def treeTest(data, treeDepth):
#     X_train, Y_train, X_test, Y_test = data
#     tree_plot = plt.subplot(111)
#     accuracy = np.empty((8), dtype='float64') # (k, L1-acc, L2-acc, L3-acc)
#
#     for depth in range(1, 8):
#         print('\n\n\n--------TESTING DEPTH-------' + str(depth))
#         tree = makeTree(X_train, Y_train, treeDepth=depth, numThresh='not implemented')
#         y_pred = testTree(tree, X_test)
#
#
#
#     return accuracy
#
# def knnVisualize(acc, width=0.2):
#     knn_plot = plt.subplot(111)
#     knn_plot.bar(acc[:,0]-width, acc[:,1], width=width, color='b', align='center', label="L1 Norm")
#     knn_plot.bar(acc[:,0], acc[:,2], width=width, color='g', align='center', label="L2 Norm")
#     knn_plot.bar(acc[:,0]+width, acc[:,3], width=width, color='r', align='center', label="L_inf Norm")
#     knn_plot.legend()
#     knn_plot.set_xlabel('k')
#     knn_plot.set_ylabel('Accuracy (%)')
#     knn_plot.set_xticks(acc[:,0])
#     knn_plot.set_title("Effects of k, L-norm on Accuracy for bagSize=1250, splitPercentage=0.7")
#     plt.show()


def treeVisualize(data, depth):
    X_train, Y_train, X_test, Y_test = data
    tree_plot = plt.subplot(111)
    accuracy = np.empty((depth), dtype='float64') # (k, L1-acc, L2-acc, L3-acc)

    for dep in range(1, depth+1):
        tree = makeTree(X_train, Y_train, treeDepth=dep, numThresh='not implemented')
        y_pred = testTree(tree, X_test)
        accuracy[dep-1] = (y_pred == Y_test).mean()

    tree_plot.bar(list(range(1, depth+1)), accuracy, color='b', align='center')
    tree_plot.set_xlabel('Tree Depth')
    tree_plot.set_ylabel('Accuracy (%)')
    tree_plot.set_xticks(list(range(1, depth+1)))
    tree_plot.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    tree_plot.set_ylim([0,1])
    tree_plot.set_title("Tree Depth vs Accuracy")
    plt.show()



def visualizeNum(data, mleRun, knnRun, bayesRun, treeRun): # assume data is 'split' type
    data_plot = plt.subplot(111)
    (X_train, Y_train, X_test, Y_test) = data

    if(mleRun): # for Q5 only
        print("Running MLE Test")
        train = np.hstack((Y_train.reshape(Y_train.shape[0], 1), X_train))
        x_values0 = train[np.where(train[:,0] == 0),1:][0]
        x_values1 = train[np.where(train[:,0] == 1),1:][0]

        (mean0, mean1, covariance_matrix0, covariance_matrix1, cp0, cp1) = mle.mle(x_values0, x_values1)
        y_pred = np.empty((X_test.shape[0]), dtype='int8')
        for i in range(X_test.shape[0]):
            y_pred[i] = mle.classifier(mean0, mean1, covariance_matrix0, covariance_matrix1, cp0, cp1, X_test[i])

        mleAcc = (y_pred == Y_test).mean()
        data_plot.bar("MLE", mleAcc, color='b', align='center')
        print(mleAcc)

    if(knnRun[0]):
        print("Running KNN Test")
        y_pred = knn(X_train, Y_train, X_test, k=knnRun[1], L=knnRun[2])
        knnAcc = (y_pred == Y_test).mean()
        data_plot.bar("KNN, k=" + str(knnRun[1]) + ", " + knnRun[2], knnAcc, color='g', align='center')
        print(knnAcc)

    if(bayesRun):
        print("Running Bayes Classifier")
        y_pred = naive_bayes(X_train, Y_train, X_test)
        bayesAcc = (y_pred == Y_test).mean()
        data_plot.bar("Bayes", bayesAcc, color='r', align='center')
        print(bayesAcc)

    if(treeRun):
        print("Running Decision Tree Classifier")
        tree = makeTree(X_train, Y_train, treeDepth=6, numThresh='not implemented')
        y_pred = testTree(tree, X_test)
        treeAcc = (y_pred == Y_test).mean()
        data_plot.bar("Decision Tree", treeAcc, color='b', align='center')
        print(treeAcc)


    data_plot.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    data_plot.set_ylim([0,1])
    data_plot.set_xlabel('Classifier')
    data_plot.set_ylabel('Accuracy (%)')
    data_plot.set_title("Accuracy of Classifiers for Split=0.7")
    plt.show()

def visualizeFairness(data, mleRun, knnRun, bayesRun, treeRun): # assume data is 'split' type
    data_plot = plt.subplot(111)
    (X_train, Y_train, X_test, Y_test) = data

    # indexArray1 = np.where(Y_test == 1)
    # indexArray0 = np.where(Y_test == 0)

    if(mleRun): # for Q5 only
        print("Running MLE Test")
        train = np.hstack((Y_train.reshape(Y_train.shape[0], 1), X_train))
        x_values0 = train[np.where(train[:,0] == 0),1:][0]
        x_values1 = train[np.where(train[:,0] == 1),1:][0]

        (mean0, mean1, covariance_matrix0, covariance_matrix1, cp0, cp1) = mle.mle(x_values0, x_values1)
        y_pred = np.empty((X_test.shape[0]), dtype='int8')
        for i in range(X_test.shape[0]):
            y_pred[i] = mle.classifier(mean0, mean1, covariance_matrix0, covariance_matrix1, cp0, cp1, X_test[i])

        mleAcc = (y_pred == Y_test).mean()
        data_plot.bar("MLE", mleAcc, color='b', align='center')
        indexArray1 = np.where(y_pred == 1)
        indexArray0 = np.where(y_pred == 0)
        print("accuracy given Y^ (pred label)=1")
        print((Y_test[indexArray1] == 1).mean())
        print("accuracy given Y^ (pred label)=0")
        print((Y_test[indexArray1] == 0).mean())


        # print(mleAcc)
        # print("accuracy for Y^ (true label)=1")
        # print((y_pred[indexArray1] == 1).mean())
        # print("accuracy for Y (true label)=0")
        # print((y_pred[indexArray0] == 0).mean())



    if(knnRun[0]):
        print("Running KNN Test")
        y_pred = knn(X_train, Y_train, X_test, k=knnRun[1], L=knnRun[2])
        knnAcc = (y_pred == Y_test).mean()
        data_plot.bar("KNN, k=" + str(knnRun[1]) + ", " + knnRun[2], knnAcc, color='g', align='center')
        # print((y_pred == 1).mean())
        # print("accuracy for Y (true label)=1")
        # print((y_pred[indexArray1] == 1).mean())
        # print("accuracy for Y (true label)=0")
        # print((y_pred[indexArray0] == 0).mean())
        indexArray1 = np.where(y_pred == 1)
        indexArray0 = np.where(y_pred == 0)
        print("accuracy given Y^ (pred label)=1")
        print((Y_test[indexArray1] == 1).mean())
        print("accuracy given Y^ (pred label)=0")
        print((Y_test[indexArray1] == 0).mean())

    if(bayesRun):
        print("Running Bayes Classifier")
        y_pred = naive_bayes(X_train, Y_train, X_test)
        bayesAcc = (y_pred == Y_test).mean()
        data_plot.bar("Bayes", bayesAcc, color='r', align='center')
        # print((y_pred == 1).mean())
        # print("accuracy for Y (true label)=1")
        # print((y_pred[indexArray1] == 1).mean())
        # print("accuracy for Y (true label)=0")
        # print((y_pred[indexArray0] == 0).mean())
        indexArray1 = np.where(y_pred == 1)
        indexArray0 = np.where(y_pred == 0)
        print("accuracy given Y^ (pred label)=1")
        print((Y_test[indexArray1] == 1).mean())
        print("accuracy given Y^ (pred label)=0")
        print((Y_test[indexArray1] == 0).mean())

    #
    # data_plot.set_xlabel('Classifier')
    # data_plot.set_ylabel('Accuracy (%)')
    # data_plot.set_title("Accuracy of Classifiers")
    # plt.show()

def visualize5():
    data = importData(returnType='split', removeRedundant=True, ignoreSensitive=False)
    visualizeNum(data, mleRun=True, knnRun=(True, 25, "L2"), bayesRun=True, treeRun=False)

def visualize6(bagLimit, splitPercentage):
    data = importData6(returnType='split', bagLimit=bagLimit, splitPercentage=splitPercentage)
    visualizeNum(data, mleRun=False, knnRun=(True, 1, "L2"), bayesRun=True, treeRun=True)


if __name__ == '__main__':
    # print(data[0][0,:])
    # data = importData6(returnType='split', bagLimit=1250, splitPercentage=0.7)
    # visualize5()
    # knnVisualize(knnTest(data, list(range(1, 10, 2))), width=0.42)
    # visualize5()
    # visualize6(5000)
    # data = importData6(returnType='split', bagLimit=5000, splitPercentage=0.7)
    # treeVisualize(data, 10)

    # (train, test) = importData(returnType='normal', removeRedundant=True, ignoreSensitive=False)
    # test_0 = test[np.where(test[:,3] == 0)] # split by race
    # test_1 = test[np.where(test[:,3] == 1)]
    #
    # data0 = (train[:,1:], train[:,0], test_0[:,1:], test_0[:,0])
    # data1 = (train[:,1:], train[:,0], test_1[:,1:], test_1[:,0])
    #
    #
    # visualizeNum(data, True, (True, 1, "L2"), True, False)
    # for x in [0.5, 0.6, 0.7, 0.8, 0.9]:
    visualize6(5000, 0.7)
