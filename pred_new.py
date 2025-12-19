# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:36:25 2024

@author: lddelnido
"""

import numpy as np
import pandas as pd
import time
from functions import * 

f1 = datetime.datetime.today()
today = greg_to_mjd(f1) 

leaps = [50082,50629,51178,53735,54831,56108,57203,57753]

mjd1,xp1,yp1,dx1,dy1,dut1 = read_iers()
xfcn, yfcn, mjd_fcn = coord_fcn(dx1, dy1, mjd1)

datea,xmassa,ymassa,zmassa,xmotiona,ymotiona,zmotiona = read_aam()
dateo,xmasso,ymasso,zmasso,xmotiono,ymotiono,zmotiono = read_oam()
dateh,xmassh,ymassh,zmassh,xmotionh,ymotionh,zmotionh = read_ham()

difa = datea[-1]-mjd1[-1]#+1
difo = dateo[-1]-mjd1[-1]#+1
difh = dateh[-1]-mjd1[-1]

if difa != 0:
    datea = datea[:-difa]
    xmassa =  xmassa[:-difa]
    ymassa = ymassa[:-difa]
    zmassa = zmassa[:-difa]
    xmotiona = xmotiona[:-difa]
    ymotiona = ymotiona[:-difa]
    zmotiona = zmotiona[:-difa]
    
if difo != 0:
    dateo = dateo[:-difo]
    xmasso =  xmasso[:-difo]
    ymasso = ymasso[:-difo]
    zmasso= zmasso[:-difo]
    xmotiono = xmotiono[:-difo]
    ymotiono = ymotiono[:-difo]
    zmotiono = zmotiono[:-difo]
    
if difh != 0:
    dateh = dateh[:-difh]
    xmassh =  xmassh[:-difh]
    ymassh = ymassh[:-difh]
    zmassh= zmassh[:-difh]
    xmotionh = xmotionh[:-difh]
    ymotionh = ymotionh[:-difh]
    zmotionh = zmotionh[:-difh]
   
xmass = [xmassa[i]+xmasso[i]+xmassh[i] for i in range(len(xmassa))]
ymass = [ymassa[i]+ymasso[i]+ymassh[i] for i in range(len(ymassa))]
zmass = [zmassa[i]+zmasso[i]+zmassh[i] for i in range(len(zmassa))]
xmotion = [xmotiona[i]+xmotiono[i]+xmotionh[i] for i in range(len(xmotiona))]
ymotion = [ymotiona[i]+ymotiono[i]+ymotionh[i] for i in range(len(ymotiona))]
zmotion = [zmotiona[i]+zmotiono[i]+zmotionh[i] for i in range(len(zmotiona))]

xsum = [xmass[i]+xmotion[i] for i in range(len(xmass))]
ysum = [ymass[i]+ymotion[i] for i in range(len(ymass))]
zsum = [zmass[i]+zmotion[i] for i in range(len(zmass))]

goal_dates = list(range(today,today+11))
pred_dates = list(range(mjd1[-1]+1,mjd1[-1]+51))
ind = pred_dates.index(goal_dates[0])

no_aam = [goal_dates]
si_aam = [goal_dates]

pred1, pred2 = pred2_dx(dx1,xfcn,xsum)    #xsum debe coincidir con las dates correspondientes
predaux1 = [item*1e3 for item in pred1[ind:ind+11]] #[as]->[mas]
predaux2 = [item*1e3 for item in pred2[ind:ind+11]]
no_aam.append(predaux1)
si_aam.append(predaux2)
 
pred1, pred2 = pred2_dy(dy1,yfcn,ysum)
predaux1 = [item*1e3 for item in pred1[ind:ind+11]]
predaux2 = [item*1e3 for item in pred2[ind:ind+11]]
no_aam.append(predaux1)
si_aam.append(predaux2)

df_no = pd.DataFrame(data = np.transpose(no_aam), columns=['MJD','dX','dY'])
df_si = pd.DataFrame(data = np.transpose(si_aam), columns=['MJD','dX','dY'])

ls = ['xpol','ypol','dUT1','LOD','dphi','deps']
for k in range(6):
    df_no.insert(loc = 1, column = ls[k], value = ['nan']*11)
    df_si.insert(loc = 1, column = ls[k], value = ['nan']*11)


np.savetxt(dd+f'/predicciones/no_eam_new/eoppcc_185_{str(today)}.txt', df_no, fmt = ['%5d','% s','% s','% s','% s','% s', '% s', '% .5f', '% .5f'],delimiter=' \t')
np.savetxt(dd+f'/predicciones/si_eam_new/eoppcc_186_{str(today)}.txt', df_si, fmt = ['%5d','% s','% s','% s','% s','% s', '% s', '% .5f', '% .5f'],delimiter=' \t')
