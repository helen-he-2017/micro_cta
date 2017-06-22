class DCC(object):
    # dynamic conditional correlations
    def __init__(self, volAdjReturns, com=200, min_periods=200):
        x = volAdjReturns.ewm(com=com, min_periods=min_periods).corr().dropna(how="all", axis=0)
        self.__matrices = {t : x.loc[t].dropna(how="all", axis=1) for t in x.index.get_level_values(level=0).unique()}

    def keys(self):
        return self.__matrices.keys()

    def __getitem__(self, item):
        return self.__matrices[item]

    def items(self):
        for time, matrix in self.__matrices.items():
            yield time, matrix