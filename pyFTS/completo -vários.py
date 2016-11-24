# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.cross_validation import KFold
# %pylab inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
from pyswarm import pso
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
from pyFTS import naive
importlib.reload(naive)


#WIN_1
path0 = r'C:\Users\Paulo-Note\Google Drive\education\engenharia UFMG\2016-2 pós\EEE933 Adv Fuzzy Time Series\sistemas nebulosos petro\data\WIN$N_1M.csv'
path1 = r'C:\Users\Paulo-Note\Google Drive\education\engenharia UFMG\2016-2 pós\EEE933 Adv Fuzzy Time Series\sistemas nebulosos petro\data\WIN$N_5M.csv'
path2 = r'C:\Users\Paulo-Note\Google Drive\education\engenharia UFMG\2016-2 pós\EEE933 Adv Fuzzy Time Series\sistemas nebulosos petro\data\WIN$N_10M.csv'
path3 = r'C:\Users\Paulo-Note\Google Drive\education\engenharia UFMG\2016-2 pós\EEE933 Adv Fuzzy Time Series\sistemas nebulosos petro\data\WIN$N_15M.csv'
# mini = pd.read_csv(path, header=0)
# mini = mini[::-1]
# WINclose = np.array(mini['close'])

#WDO
path4  = r'C:\Users\Paulo-Note\Google Drive\education\engenharia UFMG\2016-2 pós\EEE933 Adv Fuzzy Time Series\sistemas nebulosos petro\data\WDO$N_1M.csv'
path5  = r'C:\Users\Paulo-Note\Google Drive\education\engenharia UFMG\2016-2 pós\EEE933 Adv Fuzzy Time Series\sistemas nebulosos petro\data\WDO$N_5M.csv'
path6  = r'C:\Users\Paulo-Note\Google Drive\education\engenharia UFMG\2016-2 pós\EEE933 Adv Fuzzy Time Series\sistemas nebulosos petro\data\WDO$N_10M.csv'
path7  = r'C:\Users\Paulo-Note\Google Drive\education\engenharia UFMG\2016-2 pós\EEE933 Adv Fuzzy Time Series\sistemas nebulosos petro\data\WDO$N_15M.csv'
# mini= pd.read_csv(path, header=0)
# mini= mini[::-1]
# WDOclose = np.array(mini['close'])


paths=[path0,path1,path2,path3,path4,path5,path6,path7]
Wclose= np.zeros((len(paths),40000))
# Wdate= np.chararray(((len(paths),40000)))
for i in np.arange(0,len(paths)):
    mini= pd.read_csv(paths[i], header=0)
    mini= mini[::-1] #invert data, put in cronological order
    close = np.array(mini['close'])
    dates = np.array(mini['date'])
    Wclose[i][:] = close
    # Wdate[i] = dates

# 0=WIN$N_1M,
# 1=WIN$N_5M,
# 2=WIN$N_10M,
# 3=WIN$N_15M,
# 4=WDO$N_1M,
# 5=WDO$N_5M,
# 6=WDO$N_10M,
# 7=WDO$N_15M

###############################################
files=['WIN$N_1M','WIN$N_5M','WIN$N_10M','WIN$N_15M',
        'WDO$N_1M', 'WDO$N_5M','WDO$N_10M','WDO$N_15M']
i=0
size=2000 #number of elements used, maximum of 40.000
data=Wclose[i][:]
# date=Wdate[i][:]
data_used=data[len(data)-size:len(data)]
proportion = 0.9
################################################
mini= pd.read_csv(paths[i], header=0)
mini= mini[::-1] #invert data, put in cronological order
dates = np.array(mini['date'])
date_used=dates[len(dates)-size:len(dates)]

periodo_inicial= date_used[size-1]
periodo_final= date_used[0]
file_name = files[i]
#
treino = int(proportion * len(data_used))
teste = int(len(data_used) - treino)
#
data_treino = np.array(data_used[0:treino])
data_teste = np.array(data_used[treino:len(data_used)])
fig = plt.figure(figsize=[10, 5])
fig.suptitle(files[i]+' from %s to %s '%(periodo_inicial, periodo_final))
plt.plot(data_used)
print('leitura de dados ok, numero de elementos : %d '
      ', treino: %d , %d'
      ', teste: % d , %d'
      % (len(data_used),
         len(data_treino), treino ,
         len(data_teste), teste))
#
length = np.math.ceil(partitioner.avgbasedlength(data_used))
parts = partitioner.partsn(data_used)
partst= partitioner.partsn(data_treino)
data_fs_tot = partitioner.GridPartitionerTrimf(data_used, parts)
data_fs_tre = partitioner.GridPartitionerTrimf(data_treino, partst)
print("particionar ok , "
      "comprimento do intervalo : %d , "
      "numero de intervalos : %d" %(length, parts))

# Naive
naive = naive.NaiveFTS("")
naive_f = naive.forecast(data_used)
naive_ft = naive.forecast(data_teste)
# bchmk.plotComparedSeries(data_used,[cfts],[cfts_f],"blue")
print("Naive ok, forecast elements : %d , %d" %(len(naive_f),len(naive_ft)))

# CHEN
cfts = chen.ConventionalFTS("")
cfts.train(data_treino, data_fs_tre)
cfts_f = cfts.forecast(data_used)
cfts_ft = cfts.forecast(data_teste)
# bchmk.plotComparedSeries(data_used,[cfts],[cfts_f],"blue")
print("CHEN ok, forecast elements : %d , %d" %(len(cfts_f),len(cfts_ft)))

# YU
wfts = yu.WeightedFTS("")
wfts.train(data_treino, data_fs_tre)
wfts_f = wfts.forecast(data_used)
wfts_ft = wfts.forecast(data_teste)
# bchmk.plotComparedSeries(data_used,[wfts],[wfts_f],"blue")
print("YU ok, forecast elements : %d , %d" %(len(wfts_f),len(wfts_ft)))

# IWFTS
iwfts = ismailefendi.ImprovedWeightedFTS("")
iwfts.train(data_treino, data_fs_tre)
iwfts_f = iwfts.forecast(data_used)
iwfts_ft = iwfts.forecast(data_teste)
# bchmk.plotComparedSeries(data_used,[iwfts],[iwfts_f],"blue")
print("IWFTS ok, forecast elements : %d , %d"%(len(iwfts_f),len(iwfts_ft)))

# EWFTS
ewfts = sadaei.ExponentialyWeightedFTS("")
ewfts.train(data_treino, data_fs_tre, 1.1)
ewfts_f = ewfts.forecast(data_used)
ewfts_ft = ewfts.forecast(data_teste)
# bchmk.plotComparedSeries(data_used,[ewfts],[ewfts_f],"blue")
print("EWFTS ok, forecast elements : %d , %d"%(len(ewfts_f),len(ewfts_ft)))

# HOFTS
lhs_elements = 3  # LHS elements
hofts = hofts.HighOrderFTS("")
hofts.train(data_treino, data_fs_tre, lhs_elements)
hofts_f = hofts.forecast(data_used)
hofts_ft = hofts.forecast(data_teste)
# bchmk.plotComparedSeries(data_used,[hofts],[hofts_f],"blue")
print("HOFTS ok, forecast elements : %d , %d"%(len(hofts_f),len(hofts_ft)))

# DFTS
datadclose_fs = partitioner.NewGridPartitionerTrimf(data_used)
dfts = newmodel.NewDiffFTS("")
dfts.train(data_treino, data_fs_tre, datadclose_fs)
dfts_f = dfts.forecast(data_used)
dfts_ft = dfts.forecast(data_teste)
print("DFTS ok, forecast elements : %d , %d"%(len(dfts_f),len(dfts_ft)))

# COMPARATIONS
models = [naive, cfts, wfts, iwfts, ewfts, hofts, dfts]
forecasts = [naive_f, cfts_f, wfts_f, iwfts_f, ewfts_f, hofts_f, dfts_f]
forecastst = [naive_ft, cfts_ft, wfts_ft, iwfts_ft, ewfts_ft, hofts_ft, dfts_ft]
colors = ["red", "blue",'magenta', "green", "orange", "yellow", "cyan"]


from pyFTS import benchmarks as bchmk
importlib.reload(bchmk)
# for i in models: print(i)
# for i in forecasts: print(len(i))
# for i in forecastst: print(len(i))
# bchmk.compareModelsTable(data_used, models, forecasts)
bchmk.compareModelsTable(data_teste, models, forecastst)
bchmk.plotComparedSeries(data_teste, models, forecastst, colors)
#
# for i in np.arange(0,len(forecasts)): bchmk.eupdown(data_teste,forecasts[i])
for i in np.arange(0,len(forecasts)): print(bchmk.mfe(data_teste,forecastst[i]))