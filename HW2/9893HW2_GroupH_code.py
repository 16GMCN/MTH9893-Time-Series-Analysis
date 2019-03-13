"""
MTH-9893 - Homework 2
Group H: Shenyi Mao, Yueting Jiang, Chenyu Zhao, Jose Ferreira
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

"""
=======================================================================================================================
Section 1.1.
Simulate an AR(1) process with parameters alpha, beta multiple times, calculate the empirical values using MLE
for each run and calculate the average of all the runs. 
"""
def simulate_ar1_process(alpha, beta, sigma, T):
    """
    input:  alpha is the constant, beta the lagged coefficient, 
            sigma is the standard deviation and T is the number of steps
    output: empirical values of alpha, beta and sigma in a triple
    """
    # initialize an array containing the initial values of the series
    x0 = alpha / (1 - beta)
    x = np.zeros(T+1)
    x[0] = x0
    # generate a random normally distributed that represents white noise
    eps = np.random.normal(0.0, sigma, T)
    # use lagged value to calculate the i-th value
    for i in range(1, T + 1):
        x[i] = alpha + beta * x[i - 1] + eps[i - 1]
    # calculate intermediate values
    x_hat = np.mean(x[1:-1])
    x_plus = np.mean(x[2:])
    # estimate beta
    beta_hat = np.sum((x[1:-1]-x_hat)*(x[2:]-x_plus)) / np.sum((x[1:-1]-x_hat)**2)
    # estimate alpha
    alpha_hat = x_plus - beta_hat*x_hat
    # estimate sigma
    sigma_hat = np.sqrt(np.sum((x[2:]-alpha_hat-beta_hat*(x[1:-1]))**2)/T)
    return (alpha_hat, beta_hat, sigma_hat)

def run_n_ar1_simulations(alpha, beta, sigma, T, n):
    """
    input:  alpha is the constant, beta the lagged coefficient, 
            sigma is the standard deviation and T is the number of steps, n is the number of runs
    output: run the AR(1) simulation in simulate_ar1_process n times and return the average of the 
            empirical values obtained in the n runs
    """
    alphas = []
    betas = []
    sigmas = []
    # run the simulation n times, and append the MLE parameters to an array
    for i in range(n):
        (alpha_hat, beta_hat, sigma_hat) = simulate_ar1_process(alpha, beta, sigma, T)
        alphas.append(alpha_hat)
        betas.append(beta_hat)
        sigmas.append(sigma_hat)
    return (np.mean(alphas), np.mean(betas), np.mean(sigmas))


"""
=======================================================================================================================
Section 2.1.
 
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
Section containing the calls to the functions above, separated by the number of the problem they help address. 
"""

def problem_1():
    # Section 1.1: Calculate the average of MLE parameters, check if the values tend to the theoretical ones
    alpha = 0.1
    beta = 0.3
    sigma = 0.005
    N = 2000
        
    T = 100
    (alpha_hat, beta_hat, sigma_hat) = run_n_ar1_simulations(alpha, beta, sigma, T, N)
    print('T = %d'%(T))
    print('Alpha estimate = %.6f, Beta estimate = %.6f, Sigma estimate = %.6f'%(alpha_hat, beta_hat, sigma_hat))
    
    T = 250
    print('T = %d'%(T))
    (alpha_hat, beta_hat, sigma_hat) = run_n_ar1_simulations(alpha, beta, sigma, T, N)
    print('Alpha estimate = %.6f, Beta estimate = %.6f, Sigma estimate = %.6f'%(alpha_hat, beta_hat, sigma_hat))
    
    T = 1250
    print('T = %d'%(T))
    (alpha_hat, beta_hat, sigma_hat) = run_n_ar1_simulations(alpha, beta, sigma, T, N)
    print('Alpha estimate = %.6f, Beta estimate = %.6f, Sigma estimate = %.6f'%(alpha_hat, beta_hat, sigma_hat))
    

def problem_2():
    # Section 2.1: Generate time series 
    eur = pd.read_excel('HW2_Data.xlsx', sheet_name = 'EUR', 
                        index_col=0, parse_dates=True, infer_datetime_format=True)
    fed = pd.read_excel('HW2_Data.xlsx', sheet_name = 'FEDL01', 
                        index_col=0, parse_dates=True, infer_datetime_format=True)
    ecb = pd.read_excel('HW2_Data.xlsx', sheet_name = 'EUORDEPOT', 
                        index_col=0, parse_dates=True, infer_datetime_format=True)
    
    diff = fed - ecb
    
    plt.figure()
    plt.plot(eur,label='EUR')
    plt.title('EUR/USD')
    plt.xlabel('Date')
    plt.ylabel('Last')
    plt.legend()
    plt.show()
    
    plt.plot(diff,label='Rates differential')
    plt.title('Interest Rates Differential')
    plt.xlabel('Date')
    plt.ylabel('Last')
    plt.legend()
    plt.show()
    
    # run Dickey-Fuller on each series separately.
    eur_rates = eur['Last Price'].dropna()
    diff_interest_rates = diff['Last Price'].dropna()
    print("EURUSD")
    perform_adf_test(eur_rates)
    print("Diff")
    perform_adf_test(diff_interest_rates)
    
    # this should be stationary
    print("delta EURUSD")
    lagged_eur = eur['Last Price'].pct_change(1).dropna()
    perform_adf_test(lagged_eur)
    
    # this too should be stationary
    print("delta rates")
    lagged_rates = diff['Last Price'].pct_change(1).replace([np.inf, -np.inf], np.nan).dropna()
    perform_adf_test(lagged_rates)
    
    # form vector a = (1, -1)' ==> u_t = x - y
    merged_data = pd.concat([eur_rates.rename('eur'), diff_interest_rates.rename('rates')], axis=1)
    clean_data = merged_data.dropna()
    u = (clean_data['eur'] - clean_data['rates'])
    # test unit root on u, this would have to be stationary
    print("u")
    perform_adf_test(u) 
    
    

if __name__ == '__main__':
    problem_1()
    #problem_2()
    


