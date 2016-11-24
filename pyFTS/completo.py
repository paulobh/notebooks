import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.cross_validation import KFold
# %pylab inline
from pyFTS import benchmarks as bchmk
importlib.reload(bchmk)
from pyFTS import common
importlib.reload(common)
from pyFTS import partitioner
importlib.reload(partitioner)
from pyFTS import chen
importlib.reload(chen)
from pyFTS import yu
importlib.reload(yu)
from pyFTS import ismailefendi
importlib.reload(ismailefendi)
from pyFTS import hofts
importlib.reload(hofts)
from pyFTS import sadaei
importlib.reload(sadaei)
from pyFTS import newmodel
importlib.reload(newmodel)

winpath = r'C:\Users\Paulo-Note\Google Drive\education\engenharia UFMG\2016-2 pós\EEE933 Adv Fuzzy Time Series\sistemas nebulosos petro\data\WIN$N_1M.csv'
miniindex = pd.read_csv(winpath, header=0)
miniindex = miniindex[::-1]
WINclose = np.array(miniindex['close'])

WINclose = WINclose[0:20000]
proportion = 0.9
treino = int(proportion * len(WINclose))
teste = int(len(WINclose) - treino)

WIN_treino = np.array(WINclose[0:treino])
WIN_teste = np.array(WINclose[treino:len(WINclose)])
fig = plt.figure(figsize=[10, 5])
plt.plot(WINclose)

# WINdiff = common.differential(WINclose)
# fig = plt.figure(figsize=[10,5])
# plt.plot(WINdiff)

print('leitura de dados ok, numero de elementos : %d '
      ', treino: %d , %d'
      ', teste: % d , %d'
      % (len(WINclose),
         len(WIN_treino), treino ,
         len(WIN_teste), teste))

length = partitioner.avgbasedlength(WINclose)
parts = partitioner.partsn(WINclose)
# WINclose_fs = partitioner.GridPartitionerTrimf(WINclose, partitioner.partsn(WINclose))
WINclose_fs = partitioner.GridPartitionerTrimf(WIN_treino, partitioner.partsn(WIN_treino))

print("particionar ok , "
      "comprimento do intervalo : %d , "
      "numero de intervalos : %d" % (length, parts))

# CHEN
cfts = chen.ConventionalFTS("")
cfts.train(WIN_treino, WINclose_fs)
cfts_f = cfts.forecast(WINclose)
cfts_ft = cfts.forecast(WIN_teste)
# bchmk.plotComparedSeries(WINclose,[cfts],[cfts_f],"blue")
print("CHEN ok, forecast elements : %d , %d" %(len(cfts_f),len(cfts_ft)))

# YU
wfts = yu.WeightedFTS("")
wfts.train(WIN_treino, WINclose_fs)
wfts_f = wfts.forecast(WINclose)
wfts_ft = wfts.forecast(WIN_teste)
# bchmk.plotComparedSeries(WINclose,[wfts],[wfts_f],"blue")
print("YU ok, forecast elements : %d , %d" %(len(wfts_f),len(wfts_ft)))

# IWFTS
iwfts = ismailefendi.ImprovedWeightedFTS("")
iwfts.train(WIN_treino, WINclose_fs)
iwfts_f = iwfts.forecast(WINclose)
iwfts_ft = iwfts.forecast(WIN_teste)
# bchmk.plotComparedSeries(WINclose,[iwfts],[iwfts_f],"blue")
print("IWFTS ok, forecast elements : %d , %d"%(len(iwfts_f),len(iwfts_ft)))

# EWFTS
ewfts = sadaei.ExponentialyWeightedFTS("")
ewfts.train(WIN_treino, WINclose_fs, 2)
ewfts_ft = ewfts.forecast(WIN_teste)
ewfts_f = ewfts.forecast(WINclose)
# bchmk.plotComparedSeries(WINclose,[ewfts],[ewfts_f],"blue")
print("EWFTS ok, forecast elements : %d , %d"%(len(ewfts_f),len(ewfts_ft)))

# HOFTS
lhs_elements = 2  # LHS elements
hofts = hofts.HighOrderFTS("")
hofts.train(WIN_treino, WINclose_fs, lhs_elements)
hofts_f = hofts.forecast(WINclose)
hofts_ft = hofts.forecast(WIN_teste)
# bchmk.plotComparedSeries(WINclose,[hofts],[hofts_f],"blue")
print("HOFTS ok, forecast elements : %d , %d"%(len(hofts_f),len(hofts_ft)))

# DFTS
WINdclose_fs = partitioner.NewGridPartitionerTrimf(WINclose)
dfts = newmodel.NewDiffFTS("")
dfts.train(WIN_treino, WINclose_fs, WINdclose_fs)
dfts_f = dfts.forecast(WINclose)
dfts_ft = dfts.forecast(WIN_teste)
print("DFTS ok, forecast elements : %d , %d"%(len(dfts_f),len(dfts_ft)))

# COMPARATIONS
models = [cfts, wfts, iwfts, ewfts, hofts, dfts]
forecasts = [cfts_f, wfts_f, iwfts_f, ewfts_f, hofts_f, dfts_f]
forecastst = [cfts_ft, wfts_ft, iwfts_ft, ewfts_ft, hofts_ft, dfts_ft]
colors = ["red", "blue", "green", "orange", "yellow", "cyan"]

# for i in forecasts: print(len(i))
# for i in forecastst: print(len(i))
bchmk.compareModelsTable(WINclose, models, forecasts)
bchmk.compareModelsTable(WIN_teste, models, forecastst)
bchmk.plotComparedSeries(WIN_teste, models, forecastst, colors)
