import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


class Portfolio(object):
    def __init__(self, prices, cashPos, size=2e6):
        profit = (prices.pct_change() * cashPos.shift(periods=1)).sum(axis=1)

        # simulate the compounding over time
        self.__nav = (1 + profit / size).cumprod()

    @property
    def nav(self):
        return self.__nav


if __name__ == '__main__':
    fut = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True).ffill().truncate(
        before=pd.Timestamp("1990-01-01"))
    # compute volatility adjusted returns and winsorize them
    volatility = fut.pct_change().ewm(com=32).std()
    volAdjReturns = (fut.pct_change() / volatility).clip(-4.2, 4.2)

    # move back into a "price"-space by compounding those filtered returns
    prices = (1 + volAdjReturns / 100).cumprod()

    # compute the oscillator
    def osc(prices, fast=32, slow=96):
        def _mean(n): return prices.ewm(span=2 * n - 1).mean()

        def _scale(a1, a2): return 1.0 / (1/a1 + 1/a2 - 1/(a1*a2))

        return (_mean(fast) - _mean(slow)) / np.sqrt(_scale(fast, fast) - 2 * _scale(fast, slow) + _scale(slow, slow))


    p = Portfolio(prices=fut, cashPos=(50000 * np.tanh(osc(prices, fast=16, slow=48)) / volatility).clip(-5e5, 5e5))

    # simulate the compounding over time
    pct_change = p.nav.plot(logy=True, grid=True)
    plt.savefig('P2L.png')
