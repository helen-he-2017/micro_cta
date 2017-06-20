from pycta.performance.drawdown import drawdown
from pycta.performance.month import monthlytable


class Portfolio(object):
    def __init__(self, prices, cashPos, size=2e6):
        profit = (prices.pct_change() * cashPos.shift(periods=1)).sum(axis=1)

        # simulate the compounding over time
        self.__nav = (1 + profit / size).cumprod()

    @property
    def nav(self):
        return self.__nav

    @property
    def drawdown(self):
        return 100*drawdown(self.__nav)

    @property
    def monthly(self):
        return 100*monthlytable(self.__nav)