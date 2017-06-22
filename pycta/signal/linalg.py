import pandas as pd


def _ispandas(x):
    return isinstance(x, pd.DataFrame) or isinstance(x, pd.Series)


def matmul(a, b):
    """
    Matrix multiplication with Pandas DataFrames and Series
    :param a: DataFrame or Series
    :param b: DataFrame or Series
    :return:
    """
    assert _ispandas(a), "The first argument has to be a Series or a DataFrame. It is of type {0}".format(type(a))
    assert _ispandas(b), "The second argument has to be a Series or a DataFrame. It is of type {0}".format(type(b))

    # fix the order of the columns of a
    if isinstance(a, pd.DataFrame):
        inner = list(a.columns)
    else:
        inner = list(a.index)

    assert set(inner) == set(b.index), "The inner dimensions do not match"

    if isinstance(a, pd.DataFrame):
        data = a[inner].values @ b.loc[inner].values
        if isinstance(b, pd.DataFrame):
            return pd.DataFrame(data=data, index=a.index, columns=b.columns)
        if isinstance(b, pd.Series):
            return pd.Series(data=data, index=a.index)

    if isinstance(a, pd.Series):
        data = a.values.transpose() @ b.loc[inner].values
        if isinstance(b, pd.DataFrame):
            # a'*B, multiply vector from the left with matrix
            return pd.Series(data=data, index=b.columns)

        if isinstance(b, pd.Series):
            # a'*b, inner product between two vectors
            return data