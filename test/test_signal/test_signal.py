from unittest import TestCase

from pycta.signal.signal import volatility, volatility_adj_returns, returns2prices, oscillator
from test.config import read_series


class TestSignal(TestCase):
    def test_volatility(self):
        s = read_series("ts.csv", parse_dates=True)
        vola = volatility(prices=s, com=32)
        vola_adj = volatility_adj_returns(prices=s, volatility=vola, winsor=4.2)
        p = returns2prices(vola_adj)
        osc = oscillator(prices=p, fast=32, slow=96)

        self.assertAlmostEqual(vola["2015-04-22"], 0.0017237769530108324, places=10)
        self.assertAlmostEqual(vola_adj["2015-04-22"], 0.16505315982568433, places=10)
        self.assertAlmostEqual(p["2015-04-22"], 1.0377230626115819, places=10)
        self.assertAlmostEqual(osc["2015-04-22"], 0.00039868198945447524, places=10)







