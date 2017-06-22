import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


if __name__ == '__main__':
    # we recommend to use log prices instead of prices! Oscillator designed for additive prices!
    fut = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True).ffill().truncate(before=pd.Timestamp("1990-01-01"))
    # compute volatility adjusted returns and winsorize them
    volatility = np.log(fut).diff().ewm(com=32).std()
    volAdjReturns = (np.log(fut).diff() / volatility).clip(-4.2, 4.2)

    # move back into a "price"-space by compounding those filtered returns
    prices = volAdjReturns.cumsum()

    # compute the oscillator
    def osc(prices, fast=32, slow=96):
        f,g = 1 - 1/fast, 1-1/slow
        return (prices.ewm(span=2*fast-1).mean() - prices.ewm(span=2*slow-1).mean())/np.sqrt(1.0 / (1 - f * f) - 2.0 / (1 - f * g) + 1.0 / (1 - g * g))

    # compute the currency position and apply again some winsorizing to avoid extremely large positions
    CurrencyPosition = (50000*np.tanh(osc(prices, fast=16, slow=48)) / volatility).clip(-5e7, 5e7)

    # the profit today is the return today times the position of yesterday
    Profit = (fut.pct_change() * CurrencyPosition.shift(periods=1)).sum(axis=1)

    # simulate the compounding over time
    pct_change = (1 + Profit / 7e7).cumprod().plot(logy=True, grid=True)
    plt.savefig('PL.png')


    print(16*Profit.mean()/Profit.std())
    print((1 + Profit / 2e6).cumprod().resample("A").last().pct_change())
    assert False

