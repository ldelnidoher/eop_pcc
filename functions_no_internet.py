# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:05:35 2024

@author: lddelnido
"""
import numpy as np
import math
import requests
from joblib import load
from scipy.interpolate import PchipInterpolator
import numpy.linalg  as la
import copy
import os
import glob

frecFCN = -(2*math.pi)/430.0027   #Free Core Nutation frecquency
leaps = [50082,50629,51178,53735,54831,56108,57203,57753]
dd = os.getcwd()



def finals_all(start,today):
    """
    Parameters
    ----------
    mjd : int
        epoch in MJD
    today : int
        today's date at 00:00h

    Returns
    -------
    dx, dy : list of float
        solution from finals.data.iau2000 starting in start and finishing
        11 days into the future (all at 00:00h)
    fechas : list of int
        "dx", "dy" solution epochs
    """
    
    
    
    r = requests.get('https://datacenter.iers.org/data/latestVersion/finals.daily.iau2000.txt')
    rt = (r.text).split("\n")
    # f = open(dd+"/data/finals2000A.daily.txt")
    # rt = f.readlines()
    # f.close()
    ind = rt[0].index('I')
    st = int(rt[0][ind-9:ind-4])
    f = int(today-st)   # 48622 corresponds to the first epoch in the file in mjd
    i = int(start-st)
    
    lista = rt[i:f]
    lista = [lista[j].split() for j in range(len(lista))]
    xp, yp, dut1, dx, dy = [],[],[],[],[]
    
    missing_data = [] #sup missing data is at the end and not isolated
    for x in lista:
        if len(x) < 17:
            missing_data.append(1)
    v = len(missing_data)
    
    
    dx = [float(lista[k][-4])*1e-3 for k in range(len(lista))] #está en mas, no as
    dy = [float(lista[k][-2])*1e-3 for k in range(len(lista))]
    xp = [float(lista[k][-14]) for k in range(len(lista)-v)]
    yp = [float(lista[k][-12]) for k in range(len(lista)-v)]
    dut1 = [float(lista[k][-9]) for k in range(len(lista)-v)]
    fechas = [int(float(lista[k][-16])) for k in range(len(lista)-v)]
    
    xp+=[float(lista[k][-12]) for k in range(len(lista)-v,len(lista))]
    yp += [float(lista[k][-10]) for k in range(len(lista)-v,len(lista))]
    dut1 += [float(lista[k][-7]) for k in range(len(lista)-v,len(lista))]
    fechas += [int(float(lista[k][-14])) for k in range(len(lista)-v,len(lista))]
    return  fechas,xp,yp,dx,dy,dut1
    
def decimal(lista):
    """
    Parameters
    ----------
    lista : list of floats
        Coordinate in the format: [degrees, minutes, seconds]

    Returns
    -------
    int
        Tranformation of {lista} to decimal degrees

    """
    if lista[0]>=0:
        return lista[0] + lista[1] / 60 + lista[2] / 3600 
    else:
        return lista[0] - lista[1] / 60 - lista[2] / 3600


def mult(A,B):
    """
    Parameters
    ----------
    A, B : np.array of float where A.shape[1] = B.shape[0]
        Two-dimensional matrices. 

    Returns
    -------
    C : np.array
        Matrix product between A & B
    """
    C = np.zeros([A.shape[0],B.shape[1]])
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            C[i,j] = np.array(sum([A[i,r]*B[r,j] for r in range(A.shape[1])]))
    return C


def onedim(M):
    """
    Parameters
    ----------
    M : np.array of float
        Two-dimensional Hankel matrix

    Returns
    -------
    C : list
        Transforms the Hankel matrix into a one-dimensional time series
        
    Example
    -------
    In:  onedim(np.array([[0,1,2,3],[1,2,3,4],[2,3,4,5]])
    Out: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    """
    K = M.shape[1]
    L = M.shape[0]
    C = []
    for i in range(L):
        j=0
        ind = []
        while i>=0 and j<=K-1:
            ind.append(M[i,j])
            i-=1
            j+=1
        C.append(np.mean(ind))
    for j in range(1,K):
        i = L-1
        ind = []
        while i>=0 and j<=K-1:
            ind.append(M[i,j])
            i-=1
            j+=1
        C.append(np.mean(ind))
    return C



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

def read_iers():
    """
    Returns
    -------
    mjd : list of floats
        last 701 epoch values on the EOP IERS 20u23 C04 series
    xp : list of floats
        last 701 xpol values on the EOP IERS 20u23 C04 series
    yp : list of floats
        last 701 ypol values on the EOP IERS 20u23 C04 series
    dy : list of floats
        last 701 dy values on the EOP IERS 20u23 C04 series
    lod : list of floats
        last 701 lod values on the EOP IERS 20u23 C04 series
    dut : list of floats
        last 701 dut values on the EOP IERS 20u23 C04 series
    """
    r = requests.get("https://datacenter.iers.org/data/latestVersion/EOP_20u23_C04_one_file_1962-now.txt")
    datos = r.text
    
    # f = open(dd+"/data/EOP_20u23_C04_one_file_1962-now.txt")
    # datos = f.read()
    # f.close()
    cont,j = 0,0
    while cont<6:
        if datos[j] =='\n':
            cont+=1
        j+=1
    datos=datos[j:]
    ini = 50083-37665  #necesitamos los mismos datos que en el input por la fun leaps
    lista = datos.split("\n")
    aux = [lista[i].split() for i in range(len(lista)-701, len(lista)-1)]   #last value is an empty line
    aux2 = [lista[i].split() for i in range(ini, len(lista)-1)]   #last value is an empty line
    mjd = [int(float(aux[i][4])) for i in range(len(aux))]
    xp = [float(aux[i][5]) for i in range(len(aux))]
    yp = [float(aux[i][6]) for i in range(len(aux))]
    dut =  [float(aux2[i][7]) for i in range(len(aux2))]
    dx = [float(aux[i][8]) for i in range(len(aux))]
    dy = [float(aux[i][9]) for i in range(len(aux))]
    return mjd,xp,yp,dx,dy,dut


def generar(lista,k):
    date,xmass,ymass,zmass,xmotion,ymotion,zmotion=[],[],[],[],[],[],[]
    i = 0
    for i in range(0,len(lista),8):
        if k:
            aux = lista[i].split()
        else:
            aux = lista[i]
        date.append(int(float(aux[4])))
        xmass.append(float(aux[5]))
        ymass.append(float(aux[6]))
        zmass.append(float(aux[7]))
        xmotion.append(float(aux[8]))
        ymotion.append(float(aux[9]))
        zmotion.append(float(aux[10]))        
    return date,xmass,ymass,zmass,xmotion,ymotion,zmotion


def read_aam():
    """
    Returns
    -------
    epoch : list of floats
        epoch of the xmass, ymass, zmass solutions  (daily at 00:00h from 1/01/2023
                                                  up until yesterday)
    xmass : list of floats
        xmass solution of the Atmospheric Angular Momentum at said epochs
    ymass : list of floats
        idem
    zmass : list of floats
        idem
    """
    direc = dd+'/datos/AAM/'
    ls = [f'{direc}ESMGFZ_AAM_v1.0_03h_2022.asc',f'{direc}ESMGFZ_AAM_v1.0_03h_2023.asc', f'{direc}ESMGFZ_AAM_v1.0_03h_2024.asc']
    #ls = [f'{direc}ESMGFZ_AAM_v1.0_03h_2023.asc', f'{direc}ESMGFZ_AAM_v1.0_03h_2024.asc', f'{direc}ESMGFZ_AAM_v1.0_03h_2025.asc']
    date,xmass,ymass,zmass,xmotion,ymotion,zmotion = [],[],[],[],[],[],[]
    for i in range(len(ls)):
        f = open(ls[i])
        aux = (f.read()).split('\n')
        f.close()
        
        d,xma,yma,zma,xmo,ymo,zmo=generar(aux,True)
        date+=d
        xmass+=xma
        ymass+=yma
        zmass+=zma
        xmotion+=xmo
        ymotion+=ymo
        zmotion+=zmo
        

    f = open(f'{direc}ESMGFZ_AAM_v1.0_03h_2025.asc','r')
    #f = open(f'{direc}ESMGFZ_AAM_v1.0_03h_2026.asc','r')
    aamlast = f.read()
    f.close()
    
    fcast = (glob.glob(f'{direc}*F.asc'))[0]
    f = open(fcast,'r')
    aux = f.read()
    f.close()
    
    
    cont,cont2,i,j = 0,0,0,0
    
    while(cont<40):
        if aamlast[j] =="\n":
          cont+=1
        j+=1
     
        while(cont2<45):
            if aux[i] =="\n":
              cont2+=1
            i+=1
            
    aamlast=(aamlast[j:]).split("\n")
    ld = aux[i:].split("\n") #prediction of yesterday values (needed to predict today's)
    
    aamlast = [aamlast[i].split() for i in range(len(aamlast)-1)]+[ld[0].split()]
    
    d,xma,yma,zma,xmo,ymo,zmo=generar(aamlast,False)
    date+=d
    xmass+=xma
    ymass+=yma
    zmass+=zma
    xmotion+=xmo
    ymotion+=ymo
    zmotion+=zmo
    
    return date,xmass,ymass,zmass,xmotion,ymotion,zmotion

def read_oam():
    """
    Returns
    -------
    epoch : list of floats
        epoch of the xmass, ymass, zmass solutions  (daily at 00:00h from 1/01/2023
                                                  up until yesterday)
    xmass : list of floats
        xmass solution of the Oceanic Angular Momentum at said epochs
    ymass : list of floats
        idem
    zmass : list of floats
        idem
    """
    direc = direc = dd+'/datos/OAM/'
    ls = [f'{direc}ESMGFZ_OAM_v1.0_03h_2022.asc', f'{direc}ESMGFZ_OAM_v1.0_03h_2023.asc', f'{direc}ESMGFZ_OAM_v1.0_03h_2024.asc']
    # ls = [f'{direc}ESMGFZ_OAM_v1.0_03h_2023.asc', f'{direc}ESMGFZ_OAM_v1.0_03h_2024.asc',f'{direc}ESMGFZ_OAM_v1.0_03h_2025.asc']
    date,xmass,ymass,zmass,xmotion,ymotion,zmotion = [],[],[],[],[],[],[]
    for i in range(len(ls)):
        f = open(ls[i])
        aux = (f.read()).split('\n')
        f.close()
        
        d,xma,yma,zma,xmo,ymo,zmo=generar(aux,True)
        date+=d
        xmass+=xma
        ymass+=yma
        zmass+=zma
        xmotion+=xmo
        ymotion+=ymo
        zmotion+=zmo
     
    
    f = open(f'{direc}ESMGFZ_OAM_v1.0_03h_2025.asc','r')
    # f = open(f'{direc}ESMGFZ_OAM_v1.0_03h_2026.asc','r')
    oamlast = f.read()
    f.close()
    
    fcast = (glob.glob(f'{direc}*F.asc'))[0]
    f = open(fcast,'r')
    aux = f.read()
    f.close()
    
    cont,j = 0, 0
    while(cont<42):
        if oamlast[j] =="\n":
          cont+=1
        j+=1

    oamlast=(oamlast[j:]).split("\n")
    ld = aux[j:].split("\n")[2:] #prediction of yesterday values (needed to predict today's)
            
    oamlast = [oamlast[i].split() for i in range(len(oamlast)-1)]+[ld[0].split()]
    
    d,xma,yma,zma,xmo,ymo,zmo=generar(oamlast,False)
    date+=d
    xmass+=xma
    ymass+=yma
    zmass+=zma
    xmotion+=xmo
    ymotion+=ymo
    zmotion+=zmo
    
    return date,xmass,ymass,zmass,xmotion,ymotion,zmotion

def generarHAM(lista,k):
    i = 0
    date,xmass,ymass,zmass,xmotion,ymotion,zmotion = [],[],[],[],[],[],[]
    while i < len(lista):
        if k:
            aux = lista[i].split()
        else:
            aux = lista[i]
        date.append(float(aux[4]))
        xmass.append(float(aux[5]))
        ymass.append(float(aux[6]))
        zmass.append(float(aux[7]))
        xmotion.append(float(aux[8]))
        ymotion.append(float(aux[9]))
        zmotion.append(float(aux[10]))        
        i+=1
    return date,xmass,ymass,zmass,xmotion,ymotion,zmotion


def reduccionHAM(lista):  
    i = 0
    lista_aux=[]
    while i+1 < len(lista):
        lista_aux.append((lista[i]+lista[i+1])/2)
        i+=1
    return lista_aux

def read_ham():
    """
    Returns
    -------
    epoch : list of floats
        epoch of the xmass, ymass, zmass solutions  (daily at 00:00h from 1/01/2023
                                                  up until yesterday)
    xmass : list of floats
        xmass solution of the Hydrological Angular Momentum at said epochs
    ymass : list of floats
        idem
    zmass : list of floats
        idem
    """
    direc = dd+'/datos/HAM/'
    ls = [f'{direc}ESMGFZ_HAM_v1.2_24h_2022.asc',f'{direc}ESMGFZ_HAM_v1.2_24h_2023.asc',f'{direc}ESMGFZ_HAM_v1.2_24h_2024.asc']
    # ls = [f'{direc}ESMGFZ_HAM_v1.2_24h_2023.asc',f'{direc}ESMGFZ_HAM_v1.2_24h_2024.asc',f'{direc}ESMGFZ_HAM_v1.2_24h_2025.asc']
    date,xmass,ymass,zmass,xmotion,ymotion,zmotion = [59579.500],[-6.768481584344571e-08],[1.554350911101399e-07],[7.662834402983595e-10],[-5.745005402019570e-11],[-4.222078422869910e-11],[4.043618847377000e-13]
    #date,xmass,ymass,zmass,xmotion,ymotion,zmotion = [59944.500],[-1.077015111728597e-07],[1.872920064634999e-07],[9.658558932245592e-10],[-2.633075391255430e-11],[-1.347446554917640e-11],[2.536914218629560e-13]
    for i in range(len(ls)):
        f = open(ls[i])
        aux = (f.read()).split('\n')
        f.close()
        
        d,xma,yma,zma,xmo,ymo,zmo=generarHAM(aux,True)
        date+=d
        xmass+=xma
        ymass+=yma
        zmass+=zma
        xmotion+=xmo
        ymotion+=ymo
        zmotion+=zmo
    
    f = open(f'{direc}ESMGFZ_HAM_v1.2_24h_2025.asc','r')
    # f = open(f'{direc}ESMGFZ_HAM_v1.2_24h_2026.asc','r')
    hamlast = f.read()
    f.close()
    
    fcast = (glob.glob(f'{direc}*F.asc'))[0]
    f = open(fcast,'r')
    aux = f.read()
    f.close()
    
    cont,j = 0, 0
    while(cont<49):
        if hamlast[j] =="\n":
          cont+=1
        j+=1

    hamlast=(hamlast[j:]).split("\n")
    hamlast = [hamlast[i].split() for i in range(len(hamlast)-1)]
    d,xma,yma,zma,xmo,ymo,zmo=generarHAM(hamlast,False)
    date+=d
    xmass+=xma
    ymass+=yma
    zmass+=zma
    xmotion+=xmo
    ymotion+=ymo
    zmotion+=zmo
    
    date = reduccionHAM(date)
    date = [int(item) for item in date]
    xmass = reduccionHAM(xmass)
    ymass = reduccionHAM(ymass)
    zmass = reduccionHAM(zmass)
    xmotion = reduccionHAM(xmotion)
    ymotion = reduccionHAM(ymotion)
    zmotion = reduccionHAM(zmotion)
    return date,xmass,ymass,zmass,xmotion,ymotion,zmotion



def leap(x,mjd,leaps):  #entre 52000 y ahora mjd
    s = 0
    p = copy.deepcopy(x)
    for j in range(1,len(x)):
        if mjd[j-1] in leaps:
            s+=1
        p[j]-=s
    return p

def leap_inv(x, epoch,leaps,s):
    p = copy.deepcopy(x)
    p[0]+=s
    for j in range(1,len(x)):
        if epoch[j-1] in leaps:
            s+=1
        p[j]+=s
    return p

def ssa(num,param):
    """
    *** Singular Spectrum Analysis***
    
    Parameters
    ----------
    num : int
        number of the Principal Components to reconstruct the time series {param}
    param : list of floats
        time series in which we separate signal from noise by SSA

    Returns
    -------
    R : np.array of floats
        reconstructed time series using {num} principal components (signal)
    N : np.array of floats
        resiudal noise left after applying SSA to {param}
    """
    #PCA
    T = len(param)
    L = 50
    K = T-L+1
    X = np.zeros([L,K])
    #maping the {param} series into the trajectory matrix (Hankel matrix) X with a window length of L (1<L<T/2)
    for i in range(L-1):
        X[i][:] = np.array(param[(-K-L+i+1):(-L+1+i)])  
    X[-1,:] = np.array(param[-K:])        
    S = mult(X,np.transpose(X))    
    #Eigenvalues and eigenvectors
    egval,egvect = la.eig(S)        
    indices = egval.argsort()[::-1]
    egval = egval[indices]
    egvect = np.transpose((np.transpose(egvect))[indices,:])
    egvect2 = np.array([(mult(np.transpose(X),egvect[:,i].reshape(-1,1)))/math.sqrt(egval[i])
                        for i in range(X.shape[0])])
    Yaux = []
    for i in range(L):
        Yaux.append(math.sqrt(egval[i])*mult(egvect[:,i].reshape(-1,1),np.transpose(egvect2[i,:,:])))  
    Y = np.array([sum(Yaux[:num])])       #Separation in PCs for reconstruction and noise
    noise = np.array([sum(Yaux[num:])])
    F = []
    for j in range(Y.shape[0]):
        F.append(onedim(Y[j]))
    F = np.array(F)
    F2 = []
    for j in range(noise.shape[0]):
        F2.append(onedim(noise[j]))
    F2 = np.array(F2)
    R = np.array([sum(F[:,i]) for i in range(F.shape[1])])
    N = np.array([sum(F2[:,i]) for i in range(F2.shape[1])]) 
    aux = [onedim(Yaux[i]) for i in range(L)]
    for i in range(num):
        aux[i] = [aux[i][j] for j in range(len(aux[i]))]
    return R,N


def interp(fecha,R):
    """
    Parameters
    ----------
    fecha : list of floats
        epoch of each value of input {R}
    R : np.array of floats
        time series to interpolate

    Returns
    -------
    extr : np.array of floats
        extrapolation of the interpolation of R for the next 10 days
    """
    pcp = PchipInterpolator(fecha[-365:],R[-365:])
    fecha2 = [fecha[-1]-j+11 for j in range(10,0,-1)]
    extr=pcp(fecha2)
    return extr
   
 
def pred_xp(xp,xsum):
    mR,mN,m = [],[],[]
    for v in range(1,12):
        mR.append(load(dd+f'/modelos/old/no_eam/model_xp/modelR/day{v}_model_R.joblib'))   #we load the prediction models
        mN.append(load(dd+f'/modelos/old/no_eam/model_xp/modelN/day{v}_model_N.joblib'))
        m.append(load(dd+f'/modelos/old/si_eam/model_xp/day{v}_model_xp.joblib'))
    
    pred, pred_eam = [],[]
    R,N = ssa(4,xp)
    N1, N2, N3 = mN[0].n_features_in_, mR[0].n_features_in_, m[0].n_features_in_ #same features for all day models

    testN = N[-N1:]
    testR = R[-N2:]
    test_eam = list(R[-N3//3:])+list(N[-N3//3:])+list(xsum[-N3//3:])    
    
    for j in range(11):
        a = mN[j].predict(np.array(testN).reshape(1,-1))
        b = mR[j].predict(np.array(testR).reshape(1,-1))
        pred.append(float(a)+float(b))
        pred_eam.append(m[j].predict(np.array(test_eam).reshape(1,-1))[0])

    return pred, pred_eam



def pred_yp(yp,ysum):
    mR,mN,m = [],[],[]
    for v in range(1,12):
        mR.append(load(dd+f'/modelos/old/no_eam/model_yp/modelR/day{v}_model_yp.joblib'))   #we load the prediction models
        mN.append(load(dd+f'/modelos/old/no_eam/model_yp/modelN/day{v}_model_yp.joblib'))
        m.append(load(dd+f'/modelos/old/si_eam/model_yp/day{v}_model_yp.joblib'))
    
    pred, pred_eam = [],[]
    R,N = ssa(4,yp)
    N1, N2, N3 = mN[0].n_features_in_, mR[0].n_features_in_, m[0].n_features_in_ #same features for all day models
    
    testN = N[-N1:]
    testR = R[-N2:]
    test_eam = list(R[-N3//3:])+list(N[-N3//3:])+list(ysum[-N3//3:])    
    
    for j in range(11):
        a = mN[j].predict(np.array(testN).reshape(1,-1))
        b = mR[j].predict(np.array(testR).reshape(1,-1))
        pred.append(float(a)+float(b))
        pred_eam.append(m[j].predict(np.array(test_eam).reshape(1,-1))[0])

    return pred, pred_eam

    
def pred_dx(dx,xfcn,xsum):
    m1,m2 = [], []
    for v in range(1,12):
        m1.append(load(dd+f'/modelos/old/no_eam/model_dx/day{v}_model_dx.joblib'))
        m2.append(load(dd+f'/modelos/old/si_eam/model_dx/day{v}_model_dx.joblib'))
    
    N1, N2 = m1[0].n_features_in_, m2[0].n_features_in_ #same features for all day models
    
    test1 = np.array(list(dx[-N1//2:])+list(xfcn[-N1//2:])).reshape(1,-1)
    test2 = np.array(list(dx[-N2//3:])+list(xfcn[-N2//3:]+list(xsum[-N2//3:]))).reshape(1,-1)
    p1, p2 = [], []
    for j in range(11):
        p1.append(m1[j].predict(test1))
        p2.append(m2[j].predict(test2))
    p1 = (np.array(p1).transpose()).tolist()[0]
    p2 = (np.array(p2).transpose()).tolist()[0]
    return p1,p2

def pred_dy(dy,yfcn, ysum):
   m1,m2 = [], []
   for v in range(1,12):
       m1.append(load(dd+f'/modelos/old/no_eam/model_dy/day{v}_model_dy.joblib'))
       m2.append(load(dd+f'/modelos/old/si_eam/model_dy/day{v}_model_dy.joblib'))
       
   N1, N2 = m1[0].n_features_in_, m2[0].n_features_in_ #same features for all day models
   
   test1 = np.array(list(dy[-N1//2:])+list(yfcn[-N1//2:])).reshape(1,-1)
   test2 = np.array(list(dy[-N2//3:])+list(yfcn[-N2//3:]+list(ysum[-N2//3:]))).reshape(1,-1)
   p1, p2 = [], []
   for j in range(11):
       p1.append(m1[j].predict(test1))
       p2.append(m2[j].predict(test2))
   p1 = (np.array(p1).transpose()).tolist()[0]
   p2 = (np.array(p2).transpose()).tolist()[0]
   return p1,p2


def pred2_dx(dx,xfcn,xsum):  #dx, xfcn, xsum sin Bulletin A!!!!
    m1 = load(dd+'/modelos/new/day51_model_dx_fcn.joblib')
    m2 = load(dd+'/modelos/new/day51_model_dx_eam.joblib')
    N1, N2 = m1.n_features_in_, m2.n_features_in_
    test1 = np.array(list(dx[-N1//2:])+list(xfcn[-N1//2:])).reshape(1,-1)
    test2 = np.array(list(dx[-N2//3:])+list(xfcn[-N2//3:]+list(xsum[-N2//3:]))).reshape(1,-1)
    pred1 = m1.predict(test1)
    pred2 = m2.predict(test2)
    return pred1[0], pred2[0]
    
def pred2_dy(dy,yfcn,ysum):  #dy, yfcn, ysum sin Bulletin A!!!!
    m1 = load(dd+'/modelos/new/day51_model_dy_fcn.joblib')
    m2 = load(dd+'/modelos/new/day51_model_dy_eam.joblib')
    N1, N2 = m1.n_features_in_, m2.n_features_in_
    test1 = np.array(list(dy[-N1//2:])+list(yfcn[-N1//2:])).reshape(1,-1)
    test2 = np.array(list(dy[-N2//3:])+list(yfcn[-N2//3:]+list(ysum[-N2//3:]))).reshape(1,-1)
    pred1 = m1.predict(test1)
    pred2 = m2.predict(test2)
    return pred1[0], pred2[0]


def pred_dut1(dut1,zsum,mjd):
    m1,m2= [],[]
    for v in range(1,12):
        m1.append(load(dd+f'/modelos/old/no_eam/model_dut1/day{v}_model_dut1.joblib'))
        m2.append(load(dd+f'/modelos/old/si_eam/model_dut1/day{v}_model_dut1.joblib'))
        
    N1, N2 = m1[0].n_features_in_, m2[0].n_features_in_
    test1 = np.array(dut1[-N1:]).reshape(1,-1)
    test2 = np.array(list(dut1[-N2//2:])+list(zsum[-N2//2:])).reshape(1,-1)
    pred1,pred2 = [],[]
    p1a,p2a= [],[]
    for j in range(11):
        pred1.append(m1[j].predict(test1))
        pred2.append(m2[j].predict(test2))
        p1a.append((m1[j].predict(test1)).tolist()[0])
        p2a.append((m2[j].predict(test2)).tolist()[0])

    dt = list(range(mjd[-1]+1,mjd[-1]+12))
    s1 = len(list(filter(lambda n: mjd[0]<=n & n<=mjd[-1],leaps))) 

    p1 = leap_inv(p1a,dt,leaps,s1)
    p2 = leap_inv(p2a,dt,leaps,s1)
    p1 = (np.array(pred1).transpose()).tolist()[0]
    p2 = (np.array(pred2).transpose()).tolist()[0]
    return p1,p2     

def fcn(dx, dy, dt):
    """
    Parameters
    ----------
    dx : list of float
        dX solutions for epochs in {dt}
    dy : list of float
        dY solutions for epochs in {dt}
    dt : list of int
        

    Returns
    -------
    XF[0][0]: float
        Ac value for the epoch {dt[-1]+1} MJD
    XF[1][0]: float
        As value for the epoch {dt[-1]+1} MJD
        
    Note
    ----
    Calculates the amplitude of the FCN components. Used in function coord_fcn(dx,dy,dt,today)
    """
    A,D = np.zeros([400*2,2]),np.zeros([400*2,1])

    for i in range(400):
        ang = frecFCN*dt[i]
        C = math.cos(ang)
        S = math.sin(ang)
        A[2*i,0] = C
        A[2*i,1] = -S
        A[2*i+1,0] = S
        A[2*i+1,1] = C
    for i in range(400):
        D[2*i] = dx[i]
        D[2*i+1] = dy[i]
    A2 = np.transpose(A)
    C = np.linalg.inv(mult(A2,A))
    C2 = mult(A2,D)
    XF = mult(C,C2)
    
    return XF[0][0], XF[1][0]

def coord_fcn(dx,dy,dt):
    """
    Parameters
    ----------
    dx, dy : list of float
        solutions for these parameters 
    dt : list of float
        epoch of the solutions in J2000.0

    Returns
    -------
    x, y : list of floats
        solutions for xFCN and yFCN respectively
    fechas : list of int
        the corresponding epoch of the solutions {x}, {y}
    """
    x,y = np.array([]), np.array([])
    Ac, As, fechas = [],[],[]
    for i in range(400,len(dt)):
        j = i-400
        a,b = fcn(dx[j:i],dy[j:i],dt[j:i])
        Ac.append(a)
        As.append(b)
    for i in range(0,len(dt)-199):
        if i<len(Ac):
            j = i+200-1
            fechas.append(dt[j]+51544.5)
            a = dt[j]*frecFCN
            x = np.append(x,-As[i]*math.sin(a)+Ac[i]*math.cos(a))  #X_FCN para el día dt(j)
            y = np.append(y,As[i]*math.cos(a)+Ac[i]*math.sin(a)) 
        else:
            j = i+200-1
            fechas.append(dt[j]+51544.5)
            a = dt[j]*frecFCN
            x = np.append(x,-As[-1]*math.sin(a)+Ac[-1]*math.cos(a))  #X_FCN para el día dt(j)
            y = np.append(y,As[-1]*math.cos(a)+Ac[-1]*math.sin(a)) 
    fechas = [item-51544.5 for item in fechas]
    return x.tolist(),y.tolist(), fechas
