from pycta.performance.nav_series import NavSeries


class Portfolio(object):
    def __init__(self, prices, cashPos, size):
        profit = (prices.pct_change() * cashPos.shift(periods=1)).sum(axis=1)

        # simulate the compounding over time
        self.__nav = (1 + profit / size).cumprod()

    @property
    def nav(self):
        return NavSeries(self.__nav)
