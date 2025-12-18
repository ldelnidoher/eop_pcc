# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:36:25 2024

@author: lddelnido
"""

import numpy as np
import pandas as pd
import time
import math
from functions import * 
import plotly.graph_objects as go
import os

dd = os.getcwd()

f1 = datetime.datetime.today()
today = greg_to_mjd(f1) 

leaps = [50082,50629,51178,53735,54831,56108,57203,57753]

mjd1,xp1,yp1,dx1,dy1,dut1 = read_iers()
mjd2,xp2,yp2,dx2,dy2,dut2 = finals_all(mjd1[-1]+1,today)

mjd = mjd1+mjd2
xp = xp1+xp2
yp = yp1+yp2
dx = dx1+dx2
dy = dy1+dy2
dut = dut1+dut2
xfcn, yfcn, mjd_fcn = coord_fcn(dx, dy, mjd)

datea,xmassa,ymassa,zmassa,xmotiona,ymotiona,zmotiona = read_aam()
dateo,xmasso,ymasso,zmasso,xmotiono,ymotiono,zmotiono = read_oam()
dateh,xmassh,ymassh,zmassh,xmotionh,ymotionh,zmotionh = read_ham()

difa = datea[-1]-mjd[-1]
difo = dateo[-1]-mjd[-1]
difh = dateh[-1]-mjd[-1]

difa, difo = 1,1

if difa != 0:
    xmassa =  xmassa[:-difa]
    ymassa = ymassa[:-difa]
    zmassa = zmassa[:-difa]
    xmotiona = xmotiona[:-difa]
    ymotiona = ymotiona[:-difa]
    zmotiona = zmotiona[:-difa]
    
if difo != 0:
    xmasso =  xmasso[:-difo]
    ymasso = ymasso[:-difo]
    zmasso= zmasso[:-difo]
    xmotiono = xmotiono[:-difo]
    ymotiono = ymotiono[:-difo]
    zmotiono = zmotiono[:-difo]
    
if difh != 0:
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

xfcn, yfcn, mjd_fcn = coord_fcn(dx, dy, mjd)

no_aam = [list(range(today,today+11))]
si_aam = [list(range(today,today+11))]

pred1, pred2 = pred_xp(xp,xsum)
no_aam.append(pred1)
si_aam.append(pred2)

pred1, pred2 = pred_yp(yp,ysum)
no_aam.append(pred1)
si_aam.append(pred2)

pred1, pred2 = pred_dut1(dut,zsum,mjd)
no_aam.append(pred1)
si_aam.append(pred2)
 

pred1, pred2 = pred_dx(dx,xfcn,xsum)
predaux1 = [item*1e3 for item in pred1]  #[as]->[mas]
predaux2 = [item*1e3 for item in pred2]
no_aam.append(predaux1)
si_aam.append(predaux2)
 
pred1, pred2 = pred_dy(dy,yfcn,ysum)
predaux1 = [item*1e3 for item in pred1]
predaux2 = [item*1e3 for item in pred2]
no_aam.append(predaux1)
si_aam.append(predaux2)

df_no = pd.DataFrame(data = np.transpose(no_aam), columns=['MJD','xpol','ypol','dUT1','dX','dY'])
df_si = pd.DataFrame(data = np.transpose(si_aam), columns=['MJD','xpol','ypol','dUT1','dX','dY'])

ls = ['LOD','dphi','deps']
for k in range(3):
    df_no.insert(loc = 4, column = ls[k], value = ['nan']*11)
    df_si.insert(loc = 4, column = ls[k], value = ['nan']*11)
    
np.savetxt(dd+f'/predicciones/no_eam/eoppcc_167_{str(today)}.txt', df_no, fmt = ['%5d','% .8f','% .8f','% .9f','% s','% s', '% s', '% .5f', '% .5f'],delimiter=' \t')
np.savetxt(dd+f'/predicciones/si_eam/eoppcc_168_{str(today)}.txt', df_si, fmt = ['%5d','% .8f','% .8f','% .9f','% s','% s', '% s', '% .5f', '% .5f'],delimiter=' \t')
