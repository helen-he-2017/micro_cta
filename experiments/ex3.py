import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from pycta.performance.month import monthlytable
from pycta.signal import osc, volatility, volatility_adj_returns
from pycta.portfolio import Portfolio


if __name__ == '__main__':
    pd.options.display.width = 300

    fut = pd.read_csv("data/prices.csv", index_col=0, parse_dates=True).ffill().truncate(before=pd.Timestamp("1990-01-01"))

    # compute volatility adjusted returns and winsorize them
    vola = volatility(prices=fut, com=32)
    volaAdjReturns = volatility_adj_returns(prices=fut, volatility=vola, winsor=4.2)

    # move back into a "price"-space by compounding those filtered returns
    prices = (1 + volaAdjReturns / 100).cumprod()

    p = Portfolio(prices=fut, cashPos=(50000 * np.tanh(osc(prices, fast=16, slow=48)) / vola).clip(-5e5, 5e5))

    # simulate the compounding over time
    p.nav.plot(logy=True, grid=True)
    plt.savefig('P2L.png')

    print((100*monthlytable(p.nav)).to_string(float_format='{:,.2f}'.format))
