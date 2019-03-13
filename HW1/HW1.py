import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARMA

IWV = pd.read_csv("IWV.csv")
SPY = pd.read_csv("SPY.csv")

iwv = IWV["Close"]
spy = SPY["Close"]
iwv = np.array(iwv)
spy = np.array(spy)

iwv_return = (iwv[1:] - iwv[:-1]) / iwv[:-1]
spy_return = (spy[1:] - iwv[:-1]) / iwv[:-1]

diff = iwv_return - spy_return

models = []
for i in range(1, 13):
    models.append(ARMA(diff, order=(i, 0)).fit(disp=0))

aics = []
for item in models:
    aics.append(item.aic)

print(aics)

print("the p should choose: ", aics.index(min(aics))+1)
