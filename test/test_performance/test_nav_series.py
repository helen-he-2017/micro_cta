from unittest import TestCase
import pandas as pd

from pycta.performance.nav_series import NavSeries
from test.config import read_series

import pandas.util.testing as pdt
s = NavSeries(read_series("ts.csv", parse_dates=True))


class TestNavSeries(TestCase):
    def test_pos_neg(self):
        self.assertEqual(s.negative_events, 164)
        self.assertEqual(s.positive_events, 176)
        self.assertEqual(s.events, 340)

    def test_summary(self):
        pdt.assert_series_equal(s.summary().apply(str), read_series("summary.csv").apply(str))

    def test_autocorrelation(self):
        self.assertAlmostEqual(s.autocorrelation, 0.070961153249184269, places=10)

    def test_mtd(self):
        self.assertAlmostEqual(100*s.mtd, 1.4133604922211385, places=10)

    def test_ytd(self):
        self.assertAlmostEqual(100*s.ytd, 2.1718996734564122, places=10)

    def test_monthly_table(self):
        self.assertAlmostEqual(100 * s.monthlytable["Nov"][2014], -0.19540358586001005, places=5)

    def test_ewm(self):
        self.assertAlmostEqual(100 * s.ewm_volatility(periods=250).values[-1], 2.7714298334400818, places=6)
        self.assertAlmostEqual(100 * s.ewm_ret(periods=250).values[-1], 6.0365130705403685, places=6)
        self.assertAlmostEqual(s.ewm_sharpe(periods=250).values[-1], 2.1781222810347862, places=6)

    def test_fee(self):
        x = s.fee(0.5)
        self.assertAlmostEqual(x[x.index[-1]], 0.99454336215760819, places=5)
        x = s.fee(0.0)
        self.assertAlmostEqual(x[x.index[-1]], 1.0116455798589048, places=5)

    def test_monthly(self):
        self.assertAlmostEqual(s.monthly[pd.Timestamp("2014-11-30")], 1.2935771592500624, places=5)

    def test_annual(self):
        self.assertAlmostEqual(s.annual[pd.Timestamp("2014-12-31")], 1.2934720900884369, places=5)

    def test_weekly(self):
        self.assertEqual(len(s.weekly.index), 70)

    def test_truncate(self):
        x = s.truncate(before="2015-01-01")
        self.assertEqual(x.index[0], pd.Timestamp("2015-01-01"))
