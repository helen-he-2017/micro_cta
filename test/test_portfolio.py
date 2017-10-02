from unittest import TestCase

import pandas as pd

from pycta.portfolio import Portfolio
from test.config import read_frame


class TestPortfolio(TestCase):
    def test_portfolio(self):
        prices = read_frame("price_data.csv")
        cashpos = pd.DataFrame(index=prices.index, columns=prices.columns, data=50000)
        portfolio = Portfolio(prices=prices, cashPos=cashpos, size=1e6)
        self.assertAlmostEqual(portfolio.nav.sharpe_ratio(), 0.286338249832, places=10)

