from unittest import TestCase

from pycta.signal.signal import volatility, volatility_adj_returns, oscillator
from test.config import read_series


class TestSignal(TestCase):
    def test_volatility(self):
        s = read_series("ts.csv", parse_dates=True)
        vola = volatility(prices=s, com=32)
        vola_adj = volatility_adj_returns(prices=s, volatility=vola, winsor=4.2)
        osc = oscillator(prices=vola_adj.cumsum(), fast=32, slow=96)

        self.assertAlmostEqual(vola["2015-04-22"], 0.0017236025946574798, places=10)
        self.assertAlmostEqual(vola_adj["2015-04-22"], 0.16504637852209397, places=10)
        self.assertAlmostEqual(osc["2015-04-22"], 0.11165322157730379, places=10)







