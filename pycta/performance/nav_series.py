from collections import OrderedDict

import pandas as pd
import numpy as np

from .month import monthlytable
from .drawdown import drawdown as dd, drawdown_periods as dp
from .periods import period_returns, periods
from .var import value_at_risk, conditional_value_at_risk


class NavSeries(pd.Series):
    def __init__(self, *args, **kwargs):
        super(NavSeries, self).__init__(*args, **kwargs)

    @property
    def __periods_per_year(self):
        x = pd.Series(data=self.index)
        return np.round(365 * 24 * 60 * 60 / x.diff().mean().total_seconds(), decimals=0)

    def annualized_volatility(self, periods=None):
        t = periods or self.__periods_per_year
        return np.sqrt(t)*self.dropna().pct_change().std()

    @staticmethod
    def __gmean(a):
        # geometric mean A
        # Prod [a_i] == A^n
        # Apply log on both sides
        # Sum [log a_i] = n log A
        # => A = exp(Sum [log a_i] // n)
        return np.exp(np.mean(np.log(a)))

    def truncate(self, before=None, after=None):
        return NavSeries(super().truncate(before=before, after=after))

    @property
    def monthlytable(self):
        return monthlytable(self)

    @property
    def returns(self):
        return self.pct_change().dropna()

    @property
    def positive_events(self):
        return (self.returns >= 0).sum()

    @property
    def negative_events(self):
        return (self.returns < 0).sum()

    @property
    def events(self):
        return self.returns.size

    @property
    def cum_return(self):
        return (1 + self.returns).prod() - 1.0

    def sharpe_ratio(self, periods=None, r_f=0):
        return self.mean_r(periods, r_f=r_f) /self.annualized_volatility(periods)

    def mean_r(self, periods=None, r_f=0):
        # annualized performance over a risk_free rate r_f (annualized)
        periods = periods or self.__periods_per_year
        return periods*(self.__gmean(self.returns + 1.0)  - 1.0) - r_f

    @property
    def drawdown(self):
        return dd(self)

    def sortino_ratio(self, periods=None, r_f=0):
        periods = periods or self.__periods_per_year
        return self.mean_r(periods, r_f=r_f) / self.drawdown.max()

    def calmar_ratio(self, periods=None, r_f=0):
        periods = periods or self.__periods_per_year
        start = self.index[-1] - pd.DateOffset(years=3)
        # truncate the nav
        x = self.truncate(before=start)
        return NavSeries(x).sortino_ratio(periods=periods, r_f=r_f)

    @property
    def autocorrelation(self):
        """
        Compute the autocorrelation of returns
        :return:
        """
        return self.returns.autocorr(lag=1)

    @property
    def mtd(self):
        """
        Compute the return in the last available month, note that you need at least one point in the previous month, too. Otherwise NaN
        :return:
        """
        return self.resample("M").last().dropna().pct_change().tail(1).values[0]

    @property
    def ytd(self):
        """
        Compute the return in the last available year, note that you need at least one point in the previous year, too. Otherwise NaN
        :return:
        """
        return self.resample("A").last().dropna().pct_change().tail(1).values[0]

    def var(self, alpha=0.95):
        return value_at_risk(self, alpha=alpha)

    def cvar(self, alpha=0.95):
        return conditional_value_at_risk(self, alpha=alpha)

    def summary(self, alpha=0.95, periods=None, r_f=0):
        periods = periods or self.__periods_per_year

        d = OrderedDict()

        d["Return"] = 100 * self.cum_return
        d["# Events"] = self.events
        d["# Events per year"] = periods

        d["Annua. Return"] = 100 * self.mean_r(periods=periods)
        d["Annua. Volatility"] = 100 * self.annualized_volatility(periods=periods)
        d["Annua. Sharpe Ratio (r_f = {0})".format(r_f)] = self.sharpe_ratio(periods=periods, r_f=r_f)

        dd = self.drawdown
        d["Max Drawdown"] = 100 * dd.max()
        d["Max % return"] = 100 * self.returns.max()
        d["Min % return"] = 100 * self.returns.min()

        d["MTD"] = 100*self.mtd
        d["YTD"] = 100*self.ytd

        d["Current Nav"] = self.tail(1).values[0]
        d["Max Nav"] = self.max()
        d["Current Drawdown"] = 100 * dd[dd.index[-1]]

        d["Calmar Ratio (3Y)"] = self.calmar_ratio(periods=periods, r_f=r_f)

        d["# Positive Events"] = self.positive_events
        d["# Negative Events"] = self.negative_events

        d["Value at Risk (alpha = {alpha})".format(alpha=alpha)] = 100*self.var(alpha=alpha)
        d["Conditional Value at Risk (alpha = {alpha})".format(alpha=alpha)] = 100*self.cvar(alpha=alpha)
        d["First"] = self.index[0].date()
        d["Last"] = self.index[-1].date()

        return pd.Series(d)

    def ewm_volatility(self, com=50, min_periods=50, periods=None):
        periods = periods or self.__periods_per_year
        return np.sqrt(periods) * self.returns.fillna(0.0).ewm(com=com, min_periods=min_periods).std(bias=False)

    def ewm_ret(self, com=50, min_periods=50, periods=None):
        periods = periods or self.__periods_per_year
        return periods * self.returns.fillna(0.0).ewm(com=com, min_periods=min_periods).mean()

    def ewm_sharpe(self, com=50, min_periods=50, periods=None):
        periods = periods or self.__periods_per_year
        return self.ewm_ret(com, min_periods, periods) / self.ewm_volatility(com, min_periods, periods)

    @property
    def period_returns(self):
        return period_returns(self.returns, periods(today=self.index[-1]))

    def adjust(self, value=100.0):
        first = self[self.dropna().index[0]]
        return NavSeries(self * value / first)

    @property
    def monthly(self):
        return NavSeries(self.__res("M"))

    @property
    def annual(self):
        return NavSeries(self.__res("A"))

    @property
    def weekly(self):
        return NavSeries(self.__res("W"))

    def fee(self, daily_fee_basis_pts=0.5):
        ret = self.pct_change().fillna(0.0) - daily_fee_basis_pts / 10000.0
        return NavSeries((ret + 1.0).cumprod())

    @property
    def drawdown_periods(self):
        return dp(self)

    @property
    def annual_returns(self):
        x = self.annual.pct_change().dropna()
        x.index = [a.year for a in x.index]
        return x

    def __res(self, rule="M"):
        ### refactor NAV at the end but keep the first element. Important for return computations!

        a = pd.concat((self.head(1), self.resample(rule).last()), axis=0)
        # overwrite the last index with the trust last index
        a.index = a.index[:-1].append(pd.DatetimeIndex([self.index[-1]]))
        return a