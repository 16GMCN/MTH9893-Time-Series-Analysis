import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARMA
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def problem_1():
    alpha = 0.1
    beta = 0.3
    sigma = 0.005

    T1 = 100
    T2 = 250
    T3 = 1250
    N = 2000

    params = [[], [], []]

    x0 = alpha / (1 - beta)
    j = 0
    for T in [T1, T2, T3]:
        print(j)
        estimate_alpha = []
        estimate_beta = []
        estimate_sigma = []
        for i in range(N):
            x = np.zeros(T + 1)
            x[0] = x0
            eps = np.random.normal(0.0, sigma, T)
            for i in range(1, T + 1):
                x[i] = alpha + beta * x[i - 1] + eps[i - 1]
            model = ARMA(x, order=(1, 0)).fit(method='mle', disp=0)
            estimate_alpha.append(model.params[0])
            estimate_beta.append(model.params[1])
            estimate_sigma.append(np.std(model.resid))
        params[j].append(np.average(estimate_alpha))
        params[j].append(np.average(estimate_beta))
        params[j].append(np.average(estimate_sigma))
        j += 1

    print("MLE estimate parameters:")
    print("T=100: alpha: ", params[0][0], ", beta: ", params[0][1], ", sigma: ", params[0][2])
    print("T=250: alpha: ", params[1][0], ", beta: ", params[1][1], ", sigma: ", params[1][2])
    print("T=1250: alpha: ", params[2][0], ", beta: ", params[2][1], ", sigma: ", params[2][2])


'''
MLE estimate parameters:
T=100: alpha:  0.142863951757 , beta:  0.279246755367 , sigma:  0.00491525399977
T=250: alpha:  0.142846159135 , beta:  0.290040483494 , sigma:  0.0049670939894
T=1250: alpha:  0.14286352025 , beta:  0.298309954955 , sigma:  0.00499361367636

'''


def problem_2():
    data = pd.read_csv("data.csv", usecols=['FEDL01', 'EUORDEPO'])
    data.dropna(inplace=True)
    FEDL01 = data['FEDL01']
    EUORDEPO = data['EUORDEPO']

    print("test unit root in FEDL01")
    test = adfuller(FEDL01)
    print('ADF Statistic: %f' % test[0])
    print('p-value: %f' % test[1])
    for key, value in test[4].items():
        print('\t%s: %.3f' % (key, value))

    print("\ntest unit root in EUORDEPO")
    test = adfuller(EUORDEPO)
    print('ADF Statistic: %f' % test[0])
    print('p-value: %f' % test[1])
    for key, value in test[4].items():
        print('\t%s: %.3f' % (key, value))

    xt = FEDL01 - EUORDEPO

    print("\ntest unit root in FEDL01 - EUORDEPO")
    test = adfuller(xt)
    print('ADF Statistic: %f' % test[0])
    print('p-value: %f' % test[1])
    for key, value in test[4].items():
        print('\t%s: %.3f' % (key, value))


if __name__ == '__main__':
    problem_1()
    problem_2()
