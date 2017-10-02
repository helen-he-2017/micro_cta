from unittest import TestCase
import pandas as pd

import pandas.util.testing as pdt

from pycta.signal.linalg import matmul


class TestLinalg(TestCase):
    def test_matmul(self):
        a = pd.DataFrame(index=["A","B"], columns=["A","B"], data=[[1,3],[2,4]])
        b = pd.Series(index=["A","B"], data=[10,20])

        pdt.assert_series_equal(matmul(a,b), pd.Series(index=["A","B"], data=[70,100]))
        self.assertEqual(matmul(b,b), 500)
        pdt.assert_frame_equal(matmul(a,a), pd.DataFrame(index=["A","B"], columns=["A","B"], data=[[7,15],[10,22]]))
        pdt.assert_series_equal(matmul(b,a), pd.Series(index=["A","B"], data=[50, 110]))



