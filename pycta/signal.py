import numpy as np

# compute the oscillator
def osc(prices, fast=32, slow=96):
    def _mean(n): return prices.ewm(span=2 * n - 1).mean()

    def _scale(a1, a2): return 1.0 / (1 / a1 + 1 / a2 - 1 / (a1 * a2))

    return (_mean(fast) - _mean(slow)) / np.sqrt(_scale(fast, fast) - 2 * _scale(fast, slow) + _scale(slow, slow))


def volatility(prices, com=32):
    return prices.pct_change().ewm(com=com).std()

def volatility_adj_returns(prices, volatility, winsor=4.2):
    return (prices.pct_change() / volatility).clip(-winsor, winsor)