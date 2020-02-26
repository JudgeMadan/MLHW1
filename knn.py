import numpy as np
from data_import import importData

def knn(X_train, Y_train, X_test, k, L): # Returns Y_pred

    # compute distances
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    distances = np.empty((n_test, n_train))

    if(L=='L1'):
        for i in range(n_test):
            distances[i, :] = np.sum(np.abs(X_train - X_test[i, :]), axis=1)
    elif(L=='L2'):
        X_train_mag = np.sum(np.square(X_train), axis=1)
        X_test_mag = np.sum(np.square(X_test), axis=1, keepdims=True)
        prod = np.dot(X_test, X_train.T)

        distances = np.sqrt(X_test_mag - 2*prod + X_train_mag)
    elif(L=='Linf'):
        for i in range(n_test):
            distances[i, :] = np.argmax(np.abs(X_train - X_test[i, :]), axis=1)


    y_pred = np.zeros(n_test, dtype='uint8')

    for i in range(n_test): # iterate over rows
        dists_sorted = np.argsort(distances[i,:]) # sort by row
        k_nearest_labels = Y_train[dists_sorted[:k]]
        label_count = np.bincount(k_nearest_labels)
        y_pred[i] = np.argmax(label_count)

    return y_pred


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = importData(returnType='split')
    predictions = knn(X_train, Y_train, X_test, k=7, L='L2')
    print((predictions == Y_test).mean())
