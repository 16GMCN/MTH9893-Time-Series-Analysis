"""
MTH-9893 - Homework 1
Group H: Shenyi Mao, Yueting Zhang, Chenyu Zhao, Jose Ferreira
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import statsmodels.graphics.tsaplots as tsaplots
from statsmodels.tsa.ar_model import AR
import warnings
import pandas as pd

"""
=======================================================================================================================
Section 1.1.
Simulate and plot T steps of an AR(2) process with parameters alpha, beta1 and beta2 corresponding to the coefficients 
of its characteristic equation. 
"""
def simulate_ar2_process(alpha, beta1, beta2, sigma, T):
    """
    input:  alpha is the constant, beta1 and beta2 the lagged coefficients, 
            sigma is the standard deviation and T is the number of steps
    output: a plot with the simulation graph
    """
    # initialize an array containing the initial values of the series
    x0 = alpha / (1 - beta1 - beta2)
    x = np.zeros(T + 1)
    x[0] = x0
    x[1] = x0
    # generate a random normally distributed that represents white noise
    eps = np.random.normal(0.0, sigma, T)
    # use lagged values to calculate the i-th value
    for i in range(2, T + 1):
        x[i] = alpha + beta1 * x[i - 1] + beta2 * x[i - 2] + eps[i - 1]
    # plot a graph with the simulated series
    plt.figure()
    plt.plot(x, label='Xt')
    plt.title('AR(2) simulation')
    plt.xlabel('Time')
    plt.ylabel('Xt')
    plt.legend()
    plt.show()

"""
=======================================================================================================================
Section 3.1.
Retrieve daily adjusted close data for the SPY and IWV ETFs from Yahoo! Finance and calculate daily returns. 
Generate a time series with the difference SPY - IWV. 
"""

def plot_price_series(spy, iwv):
    """
    input:  two timeseries representing the SPY and IWV prices
    output: a plot with both series
    """
    plt.figure()
    plt.plot(spy,label='SPY')
    plt.plot(iwv,label='IWV')
    plt.title('SPY/IWV prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    
def calculate_diff_returns(spy, iwv):
    """
    input:  two timeseries representing the SPY and IWV prices
    output: add a daily returns column to the original time series and return the difference between the two returns 
    """
    spy.loc[:, 'Returns'] = spy['Adj Close'].pct_change(1)
    iwv.loc[:, 'Returns'] = iwv['Adj Close'].pct_change(1)
    xt = (spy['Returns'] - iwv['Returns']).dropna()
    return xt

def plot_diff(diff):
    """
    input:  a timeseries representing the daily difference between SPY and IWV returns
    output: add a daily returns column to the original time series and calculate the difference between the two 
    """
    plt.figure()
    plt.plot(diff,label='Xt')
    plt.title('Difference of returns (SPY-IWV)')
    plt.xlabel('Time')
    plt.ylabel('Difference')
    plt.legend()
    plt.show()


"""
=======================================================================================================================
Section 3.2.
Perform simple checks to make sure the series is stationary.
Split the data in 3 contiguous sections and calculate the mean and variance of each group. 
A stationary series should have similar means and variances for all its sections.
"""

def calculate_split_stats(xt):
    """
    input:  a timeseries representing the daily difference between SPY and IWV returns
    output: prints means and variances of three sections of the series  
    """
    size = len(xt)
    # split the series in 3 sections
    x1, x2, x3 = xt[0:size//3], xt[(size//3)+1:(2*size)//3], xt[((2*size)//3)+1:]
    # calculate means and variances for all sections to check if they are roughly the same
    mean1, mean2, mean3 = x1.mean(), x2.mean(), x3.mean()
    var1, var2, var3 = x1.var(), x2.var(), x3.var()
    print('mean1=%.6f, mean2=%.6f, mean3=%.6f' % (mean1, mean2, mean3))
    print('variance1=%.6f, variance2=%.6f, variance3=%.6f' % (var1, var2, var3))    
    
"""
=======================================================================================================================
Section 3.3.
Another check for stationarity: Generate the auto-correlation function and partial auto-correlation function plots
and analyze
"""

def plot_acf_pacf(xt, n):
    """
    input:  a timeseries representing the daily difference between SPY and IWV returns and a number of lag values to
            be displayed
    output: outputs the acf and pacf plots of the time series passed as input  
    """
    tsaplots.plot_acf(xt,lags=n)
    tsaplots.plot_pacf(xt,lags=n)

"""
=======================================================================================================================
Section 3.4.
Another check for stationarity: Perform an statistic test for stationarity, the augmented Dickey-Fuller test.
"""

def perform_adf_test(xt):
    """
    input:  a timeseries representing the daily difference between SPY and IWV returns
    output: outputs the statistic values from the ADF test  
    """
    test = adfuller(xt)
    print('ADF Statistic: %f' % test[0])
    print('p-value: %f' % test[1])
    for key, value in test[4].items():
        print('\t%s: %.3f' % (key, value))

"""
=======================================================================================================================
Section 3.5.
Fit the series to an AR(p) model
"""

def estimate_p(xt, max_lags):
    """
    input:  a timeseries representing the daily difference between SPY and IWV returns and a number of maximum lags
            to attempt to incorporate in the fitting process
    output: outputs the AIC values from testing with all numbers of lagged values up to max_lags
    """
    print('| # Lags\t| AIC\t\t|')
    print('--------------------------------')
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        for i in range(1, max_lags+1):
            model=AR(xt).fit(maxlag=i,ic='aic',method='mle')
            print('| ' + str(i) + '\t\t| ' + str(model.aic) + '|')

def fit_ar(xt, lags):
    """
    input:  a timeseries representing the daily difference between SPY and IWV returns and a number of lags
            to use in the fitting process
    output: outputs the fit coefficients calculated using maximum likelihood estimation
    """
    print('Fitting time series using %d lags...'%(lags))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        model = AR(xt).fit(maxlag=lags,ic='aic',method='mle')
    print(model.params)
    
"""
=======================================================================================================================
Section containing the calls to the functions above, separated by the number of the problem they help address. 
"""
def problem_1():
    # Section 1.1: Simulate the time series given in the problem 
    alpha = 1.92 
    beta1 = -1.1
    beta2 = 0.18
    sigma = 0.001
    T = 100 
    simulate_ar2_process(alpha, beta1, beta2, sigma, T)

def problem_3():
    # Section 3.1: Generate time series of daily return differences 
    sp = pd.read_csv('SPY.csv', usecols=['Date','Adj Close'], index_col=0, parse_dates=True, infer_datetime_format=True)
    iw = pd.read_csv('IWV.csv', usecols=['Date','Adj Close'], index_col=0, parse_dates=True, infer_datetime_format=True)
    plot_price_series(sp, iw)
    xt = calculate_diff_returns(sp, iw)
    plot_diff(xt)
    # Section 3.2: Stationarity check: means and variances remain stable across the entire time series
    calculate_split_stats(xt)
    # Section 3.3: Stationarity check: ACF and PACF plots
    lags = 60 #plot only the first 60 auto-correlation values
    plot_acf_pacf(xt, lags)
    # Section 3.4: Stationarity check: augmented Dickey-Fuller test
    perform_adf_test(xt)
    # Section 3.5: Estimate parameters and find fit coefficients
    # we chose the maximum number of lags to be 20, it's a number that we thought might achieve a balance 
    # between simplicity and power of prediction. 
    max_lags = 20
    estimate_p(xt, max_lags)
    fit_ar(xt,3)
    fit_ar(xt,20)

if __name__ == '__main__':
    problem_1()
    problem_3()


