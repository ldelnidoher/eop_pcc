# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 09:31:18 2026

@author: lddelnido
"""

import numpy as np
from astropy.time import Time
import os
import shutil
import sqlite3
from contextlib import closing
import webbrowser
from platformdirs import user_downloads_dir
import time
import zipfile
import sys

# Path to app local repository
direc2 = "C:/Users/becario.adsaz/Documents/EOP_Predictions/stream_app"

################### AUXILIARY  FUNCTIONS ################################
def read_db(num,lista):
    with closing(sqlite3.connect(f"{direc2}/eop_predictions.db")) as conn:
        with closing(conn.cursor()) as cursor:
            cursor.execute(f""" {to_str(lista,num)}  """)
        conn.commit()   
    return 0

"""
SI ERROR AL EJECUTAR CON RESTART EL KERNEL SUELE FUNCIONAR
"""

def to_str(lst,num):
    aa = ['eop_old','eop_new','fcn_cpo']
    st = f'INSERT INTO {aa[num]} (date, epoch, ac, [as], x0, y0, dx, dy) VALUES '
    for x in lst:
        st = st+str(tuple(x))+','
    return st[:-1]+';'

def delete_range(conn, id_inicio, id_fin, num):
    aa = ['eop_old','eop_new','fcn_cpo']
    query = f"DELETE FROM {aa[num]} WHERE epoch BETWEEN ? AND ?"
    with closing(conn.cursor()) as cursor:
        cursor.execute(query, (id_inicio, id_fin))
    conn.commit()

def get_last_epoch(conn, num, param):
    aa = ['eop_old','eop_new','fcn_cpo']
    query = f"SELECT MAX({param}) FROM {aa[num]}"
    with closing(conn.cursor()) as cursor:
        cursor.execute(query)
        result = cursor.fetchone()
    return result[0]
            
###################### FCN ###########################

# Download file .zip from repository
webbrowser.open("https://unialicante-my.sharepoint.com/:u:/g/personal/santiago_belda_mscloud_ua_es/IQDmLGAPUE3CSbJmOpiesTZ6AZ9r3QPzk0kZfidSxidSllY?e=rRhHk6&download=1", new=2)
time.sleep(10)

# Get downloads folder
downloads = user_downloads_dir()
print(downloads+'\CPO_FCN.zip')

# Extract folder from .zip
with zipfile.ZipFile(downloads+'\CPO_FCN.zip', 'r') as z:
    z.extractall(downloads+'\CPO_FCN')

# Get last day of last prediction and delete all predicted data from last update
with closing(sqlite3.connect(f"{direc2}/eop_predictions.db")) as conn:
    last_day=get_last_epoch(conn, 2, 'epoch')
    delete_range(conn, last_day-564, last_day, 2) # Remove last predicted data

# Today
mjd = int(Time.now().mjd)
fday = (Time(mjd, format = 'mjd').to_datetime()).strftime('%Y%m%d')

# Get today's filename
dir1 = downloads+f'\CPO_FCN\FCN_CPO_IERS_C0420_{fday}.txt'
print(dir1)

# Open file
try :
    f= open(dir1,'r')
    data = f.readlines()
    f.close()
except FileNotFoundError:
    # If files are not updated, delete .zip and unpacked folder
    print("No se ha encontrado el fichero actual")
    file = os.path.join(downloads, "CPO_FCN.zip")
    directory = os.path.join(downloads, "CPO_FCN")
    if os.path.exists(file):
        os.remove(file)
        print("Archivo eliminado")
    else:
        print("El archivo no existe")
    if os.path.exists(directory):
        shutil.rmtree(directory)
        print("Archivo eliminado")
    else:
        print("El archivo no existe")
    sys.exit()

# Get last day of new predictions
new_day=int(float(data[-1].split()[0]))

# Copy only needed data for the update
data = list(np.transpose([j.split() for j in data[(last_day-565-new_day):]]))

# Adapt data to database format
data[0] = [int(float(x)) for x in data[0]]
for j in range(1,7):
    data[j] = [float(x) for x in data[j]]
rows = []
a= data[0]
mjd = [Time(j,format = 'mjd') for j in a]
date = [j.iso[:-4] for j in mjd]
rows.append(date)
for j in range(0,7):
    rows.append(data[j])
rows1 = [[fila[i] for fila in rows] for i in range(len(rows[0]))]

# Update database with new data
df_old= read_db(2,rows1)

# Delete .zip and unpacked folder
file = os.path.join(downloads, "CPO_FCN.zip")
directory = os.path.join(downloads, "CPO_FCN")
if os.path.exists(file):
    os.remove(file)
    print("Archivo eliminado")
else:
    print("El archivo no existe")
if os.path.exists(directory):
    shutil.rmtree(directory)
    print("Archivo eliminado")
else:
    print("El archivo no existe")

