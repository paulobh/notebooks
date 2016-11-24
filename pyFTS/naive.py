import numpy as np
from pyFTS import common
from pyFTS import fts


class NaiveFLRG:
    def __init__(self, LHS):
        self.LHS = LHS
        self.RHS = set()

    def append(self, c):
        self.RHS.add(c)

    def __str__(self):
        tmp = self.LHS.name + " -> "
        tmp2 = ""
        for c in sorted(self.RHS, key=lambda s: s.name):
            if len(tmp2) > 0:
                tmp2 += ","
            tmp2 = tmp2 + c.name
        return tmp + tmp2


class NaiveFTS(fts.FTS):
    def __init__(self, name):
        super(NaiveFTS, self).__init__(1, "Naive")
        self.name = "Naive FTS"
        self.detail = "Naive"
        self.flrgs = {}

    # def generateFLRG(self, flrs):
    #     flrgs = {}
    #     for flr in flrs:
    #         if flr.LHS.name in flrgs:
    #             flrgs[flr.LHS.name].append(flr.RHS)
    #         else:
    #             flrgs[flr.LHS.name] = ConventionalFLRG(flr.LHS);
    #             flrgs[flr.LHS.name].append(flr.RHS)
    #     return (flrgs)

    # def train(self, data, sets):
    #     self.sets = sets
    #     tmpdata = common.fuzzySeries(data, sets)
    #     flrs = common.generateNonRecurrentFLRs(tmpdata)
    #     self.flrgs = self.generateFLRG(flrs)

    def forecast(self, data):
        ndata = np.array(data)
        l = len(ndata)
        ret = []
        for k in np.arange(0, l):
            ret.append(ndata[k])
        return np.array(ret)

