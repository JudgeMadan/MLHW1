import numpy as np
from data_import import importData

def naive_bayes(x_train, y_train, x_test):

    unique, counts = np.unique(y_train, return_counts=True)
    d = dict(zip(unique, counts))


    probability = counts/np.sum(counts)
    num_classes = probability.shape[0]
    num_features = x_train.shape[1]


    classList = []
    for i in range(num_classes):
        featureList = []
        X_class_i = np.take(x_train, np.where(y_train == unique[i])[0], axis=0)
        for ft in range(num_features):
            unq, cnt = np.unique(X_class_i.T[ft], return_counts=True)
            cnt = cnt/X_class_i.shape[0]
            featureList.append(dict(zip(unq, cnt)))

        classList.append(featureList)


    y_pred = np.empty((x_test.shape[0]), dtype='int8')
    i = 0
    for x in x_test:
        prob = np.ones((num_classes), dtype='float64')
        for cl in range(num_classes):
            for ft in range(num_features):
                prob[cl] = prob[cl] * classList[cl][ft].get(x[ft], 0)
        prob = prob * probability # probability = class prior
        y_pred[i] = unique[np.argmax(prob)]
        i += 1

    return y_pred


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = importData(returnType='split')
    y_pred = naive_bayes(X_train, Y_train, X_test)
    print((y_pred == Y_test).mean())
