import math
import numpy as np
from pyFTS import common

def avgbasedlength(data):
    datadiff = common.differential(data)
    datadiff_avg = np.mean(np.abs(datadiff)) / 2
    if datadiff_avg >= 0.1 and datadiff_avg <= 1:
        base = 0.1
    elif datadiff_avg > 1 and datadiff_avg <= 10:
        base = 1
    elif datadiff_avg > 10 and datadiff_avg <= 100:
        base = 10
    elif datadiff_avg > 100 and datadiff_avg <= 1000:
        base = 100
    else:
        base = 1000
    if base<1:
        length = round(datadiff_avg, int(-math.log10(base)))
    else:
        length = round(datadiff_avg, int(-math.log10(base)+1))
    return length


def partsn(data):
    Dmax = round(max(data), -2)
    Dmin = round(min(data), -2)
    return int(round((Dmax - Dmin) / avgbasedlength(data)))

def GridPartitionerTrimf(data, npart , names=None, prefix="A"):
    sets = []
    dmax = max(data)
    dmax = dmax + dmax * 0.10
    dmin = min(data)
    dmin = dmin - dmin * 0.10
    dlen = dmax - dmin
    partlen = math.ceil(dlen / npart)
    partition = math.ceil(dmin)
    for c in range(npart):
        sets.append(common.FuzzySet(prefix + str(c), common.trimf,
                                    [round(partition - partlen, 3), partition, partition + partlen], partition))
        partition = partition + partlen
    return sets


def NewGridPartitionerTrimf(data, names=None, prefix="DU"):
    sets = []
    Dmax = round(max(data), -len(str(max(data))) + 2)
    Dmin = round(min(data), -len(str(min(data))) + 2)
    # dlen = Dmax - Dmin
    m = partsn(data)
    npart = 2 * (m - 1) + 1
    # partlen = math.ceil(dlen / npart)
    partlen = 1
    # partition = math.ceil(dmin)
    partition = -(m - 1)
    for c in range(npart):
        sets.append(common.FuzzySet(prefix + str(-(m-1)+c),
                                    common.trimf,
    [round(partition-partlen,3), partition, partition+partlen],
                                    partition))
        partition = partition + partlen
    return sets
