import numpy as np
from pyFTS import common
from pyFTS import fts
from pyFTS import partitioner
from pyFTS import benchmarks as bchmk
from pyswarm import pso

class NewDiffFLRG:
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
                tmp2 = tmp2 + ","
            tmp2 = tmp2 + c.name
        return tmp + tmp2

class NewDiffFTS(fts.FTS):
    def __init__(self, name):
        super(NewDiffFTS, self).__init__(1, "DFTS")
        self.name = "Diff FTS"
        self.detail = "new model prof"

    def generateFLRG(self, flrs):
        flrgs = {}
        for flr in flrs:
            if flr.LHS.name in flrgs:
                flrgs[flr.LHS.name].append(flr.RHS)
            else:
                flrgs[flr.LHS.name] = NewDiffFLRG(flr.LHS);
                flrgs[flr.LHS.name].append(flr.RHS)
        return flrgs

    def train(self, data, sets, dsets):
        self.sets = sets
        self.dsets = dsets
        tmpdata = common.DFS(data, sets, dsets)
        flrs = common.generateRecurrentFLRs(tmpdata)
        self.flrgs = self.generateFLRG(flrs)

    def forecast(self, data,xopt):
        ndata = np.array(data)
        l = len(ndata)
        delaydiff=1
        ret = []
        [alpha, beta] = xopt
        if l <= delaydiff:
            return data
        for k in np.arange(delaydiff, l):
            mv = common.fuzzyInstance(ndata[k-delaydiff], self.sets)
            actual = self.sets[np.argwhere(mv == max(mv))[0, 0]]
            dmv = common.fuzzyInstance(ndata[k], self.dsets)
            dactual = self.dsets[np.argwhere(dmv == max(dmv))[0, 0]]
            if dactual.name not in self.flrgs:
                ret.append(actual.centroid)
            else:
                flrg = self.flrgs[dactual.name]
                mp = self.getMidpoints(flrg)
                length= partitioner.avgbasedlength(data)
                ret.append(alpha*ndata[k]+beta*(sum(mp)/len(mp))*length)
        if len(ret)<len(data):
            for i in np.arange(0, len(data)-len(ret)):
                ret.insert(i, data[i])
        return ret

    # def pso(self, data, lb, ub, f_ieqcons):
    #     forecasts = self.forecast(data,xopt)
    #     error_r = bchmk.rmse(data,forecasts)
    #     alpha
    #     return xopt
    def error(self,x,*args):
        # alpha, beta=x
        data = args
        e = bchmk.rmse(data,self.forecast(data,x))
        return e

    def pso_opt(self, *args):
        data = args
        lb = [0, 0]
        ub = [1, 1]
        xopt, fopt = pso(self.error(),lb,ub,
                     f_ieqcons=forecast(data,xopt),args=data)