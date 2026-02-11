# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 09:31:18 2026

@author: lddelnido
"""

import numpy as np
import pandas as pd
import math
import time
import datetime 
from astropy.time import Time
import os
import matplotlib.pyplot as plt
import sqlite3

day = 61082

#Path to app local repository
direc2 = "C:/Users/lddelnido/Documents/app/stream_app/"

################### AUXILIARY  FUNCTIONS ################################
def read_db(num,lista):
    table = ['eop_old','eop_new']
    conn = sqlite3.connect(f"{direc2}/eop_predictions.db")
    cursor = conn.cursor()
    cursor.execute(f""" {to_str(lista,num)}  """)
    conn.commit()
    conn.close()
    return 0

"""
SI ERROR AL EJECUTAR CON RESTART EL KERNEL SUELE FUNCIONAR
"""

def greg_to_mjd(f):   
    """
    Parameters
    ----------
    f : datetime.datetime
        Gregorian time date

    Returns
    -------
    mjd : int
        f at 00:00h in MJD
    """
    y,m,d=f.timetuple()[:3]
    jd = 367*y-int((7*(y+int((m+9)/12.0)))/4.0)+int((275*m)/9.0)+d+1721013.5-0.5*math.copysign(1,100*y+m-190002.5)+0.5
    mjd = int(jd-2400000.5)
    return mjd

def to_str(lst,num):
    aa = ['eop_old','eop_new']
    st = f'INSERT INTO {aa[num]} (pub_date,parameter,type_eam,pred_day0, pred_day1, pred_day2, pred_day3, pred_day4, pred_day5, pred_day6, pred_day7, pred_day8, pred_day9, pred_day10) VALUES '
    for x in lst:
        st = st+str(tuple(x))+','
    return st[:-1]+';'
            
###################### EOP_OLD & NEW ###########################

d = Time(day,format = 'mjd')
d.format = 'iso'
d = d.value[:-4]


direc = os.getcwd()+'/predicciones/'
dirs1 = [f'{direc}no_eam/eoppcc_167_{day}.txt',
        f'{direc}si_eam/eoppcc_168_{day}.txt']
dirs2 = [f'{direc}no_eam_new/eoppcc_185_{day}.txt',
        f'{direc}si_eam_new/eoppcc_186_{day}.txt']

for i in range(2):
    f = open(dirs1[i],'r')
    data = f.readlines()
    f.close()
    
    data = list(np.transpose([j.split() for j in data]))
    
    names = ['epoch','xpol','ypol','dut1','dx','dy']
    num = [0,1,2,3,7,8]
    
    data[0] = [int(x) for x in data[0]]
    for j in range(1,6):
        data[num[j]] = [float(x) for x in data[num[j]]]
      
    
    rows1 = []
    for j in range(len(names)):
        aux = [d,names[j],i]
        if num[j] == 0:
            aux = aux+data[num[j]]
        else:
            aux = aux+ data[num[j]]
        rows1.append(aux)
     
    df_old= read_db(0,rows1)
    
    

    f = open(dirs2[i],'r')
    data = f.readlines()
    f.close()

    data = list(np.transpose([j.split() for j in data]))

    names = ['epoch','dx','dy']
    num = [0,7,8]

    data[0] = [int(x) for x in data[0]]
    for j in range(1,3):
        data[num[j]] = [float(x) for x in data[num[j]]]
        
    rows2 = []
    for j in range(len(names)):
        aux = [d,names[j],i]
        if num[j] == 0:
            aux = aux+data[num[j]]
        else:
            aux = aux+ data[num[j]]
        rows2.append(aux)
     
    df_new = read_db(1,rows2)
    

