import numpy as np


# compute the oscillator
def oscillator(prices, fast=32, slow=96):
    assert fast < slow, "fast >= slow!"

    def _mean(n): return prices.ewm(span=2 * n - 1).mean()

    def _inv(x): return 1.0 / x

    def _scale(a1, a2): return _inv(_inv(a1) + _inv(a2) - _inv(a1 * a2))

    return (_mean(fast) - _mean(slow)) / np.sqrt(_scale(fast, fast) - 2 * _scale(fast, slow) + _scale(slow, slow))


def volatility(prices, com=32, min_periods=0):
    return prices.pct_change().ewm(com=com, min_periods=min_periods).std()


def volatility_adj_returns(prices, volatility, winsor=4.2):
    return (prices.pct_change() / volatility).clip(-winsor, winsor)


def returns2prices(returns):
    return (1 + returns/100).cumprod()


