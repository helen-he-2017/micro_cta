import os
import pandas as pd


def resource(name):
    base_dir = os.path.dirname(__file__)
    return os.path.join(base_dir, "resources", name)


def read_frame(name, parse_dates=True, index_col=0):
    return pd.read_csv(resource(name), index_col=index_col, header=0, parse_dates=parse_dates)


def read_series(name, parse_dates=True, index_col=0, column_name=None):
    return pd.read_csv(resource(name), index_col=index_col, header=None, squeeze=True, parse_dates=parse_dates, names=[column_name])
