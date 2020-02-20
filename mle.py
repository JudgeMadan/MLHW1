import numpy as np
from data_import import importData
from scipy.stats import multivariate_normal






def mle(x_values0, x_values1):
    covariance_matrix0 = np.cov(x_values0, rowvar = False)
    covariance_matrix1 = np.cov(x_values1, rowvar = False)
    mean0 = x_values0.mean(axis=0)
    mean1 = x_values1.mean(axis=0)

    return (mean0, mean1, covariance_matrix0, covariance_matrix1)

def classifier(mean0, mean1, cov0, cov1, test):
    gaussian0 = multivariate_normal(mean0, cov0)
    gaussian1 = multivariate_normal(mean1, cov1)

    return 0 if gaussian0.pdf(test) > gaussian1.pdf(test) else 1


if __name__ == '__main__':
    (x_values0, x_values1, x_test, y_test) = importData(returnType='mle', ignoreSensitive=False)
    (mean0, mean1, covariance_matrix0, covariance_matrix1) = mle(x_values0, x_values1)
    y_pred = np.empty((x_test.shape[0]), dtype='int8')
    for i in range(x_test.shape[0]):
        y_pred[i] = classifier(mean0, mean1, covariance_matrix0, covariance_matrix1, x_test[i])
    print((y_pred == y_test).mean())
