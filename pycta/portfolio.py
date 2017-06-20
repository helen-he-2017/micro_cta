class Portfolio(object):
    def __init__(self, prices, cashPos, size=2e6):
        profit = (prices.pct_change() * cashPos.shift(periods=1)).sum(axis=1)

        # simulate the compounding over time
        self.__nav = (1 + profit / size).cumprod()

    @property
    def nav(self):
        return self.__nav
