from unittest import TestCase

import pandas as pd

from pycta.signal.dcc import DCC
from test.config import read_frame
from pycta.signal.signal import volatility_adj_returns, volatility


class TestDCC(TestCase):
    def test_dcc(self):
        frame = read_frame(name="price_data.csv")
        vola = volatility(prices=frame)
        volaAdj = volatility_adj_returns(prices=frame, volatility=vola)

        dcc = DCC(volAdjReturns=volaAdj)
        mat = dcc[pd.Timestamp("2013-10-09")]
        self.assertAlmostEqual(mat["G"]["A"], 0.094022481720826195, places=10)


