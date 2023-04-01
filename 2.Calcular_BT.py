# %%
import numpy as np  # Operaciones matemáticas y manejo de matrices
import matplotlib.pyplot as plt  # Gráficos
import pandas as pd  # Manejo de fechas
import os
import glob
from netCDF4 import Dataset
from BT import *

'''
Este código hace uso de la librería BT. Aquí se importan las funciones y clases
creadas necesarias para calcular las retrotrayectorias.

Es necesario cambiar las rutas de los archivos, las fechas de cálculo y las
coordenas de inicio del cálculo de las retrotrayectorias
'''


# Ruta donde están almacenados los datos de ERA5
path_files = 'datos/'

# Ruta donde se van a guardar las retrotrayectorias por día.
path_out = 'procesados/'

# Coordenadas de inicio del cálculo
lati = 4.60971
loni = -74.08175

# Nivel de inicio del cálculo
level = [825]  # hPa

# Propiedades del cálculo
ndays = 10  # cantidad de días en los que se va a hacer el cálculo
delta_t = 6   # horas

# Fechas en las que las que se va a realizar el cálculo
# Estas son las fechas que se deben cambiar para las fechas de estudio deseadas
timerange = pd.date_range('1980-01-15', '1980-01-20', freq='d')

# Se carga el objeto db (database) con los datos necesarios para realizar
# los c+alculos. Las porpiedades se extraen de los archivos leidos
db = Follow_ERA5(path=path_files, lati=lati, loni=loni)

# Se pueden consultar los archivos .nc en esta DataFrame
db.data_base

# el cálculo de las BT se va a relizar cada para cada día
# Por ejempl se agarra el 15 de enero de 1980, para este día se van a
# calcular 4 retrotrayectorias, iniciadas independientemente a las
# 00:00, 06:00, 12:00 y 18:00, con una resolución temporal de 6 horas y un
# tiempo de viaje de 10 días.
for i, date_i in enumerate(timerange):
    
    for level_i in level:
        # Nombre del archivo, correspondiente al día evaluado
        date_name = f'{date_i.strftime("%Y%m%d")}'

        # se limitan un solo día de calculo
        fechai = date_i.strftime('%Y-%m-%d 00:00')
        fechaf = date_i.strftime('%Y-%m-%d 23:59')
        print(f'{fechai} -----> {fechaf}')
        name_file = f'BT.{delta_t}h.{level_i}hPa.{date_name}.nc'

        # Se calculan las retrotrayectorias para un día particular
        db.Trajectories_level_i(fechai=fechai, fechaf=fechaf, delta_t=delta_t,
                                ndays=ndays, level_i=level_i)

        # Para un día particular el diccionario BT cotiene la información
        # espacial de las 4 retrotrayectorias
        db.BT

        # Se puede graficar la retrotrayectoria de un día particular
        db.Plot_Trajectories(plot_scatter=True)

        file_out = f'{path_out}{name_file}'

        # Se guarda un archivo BT con la información de las retrotrayectorias
        if not os.path.exists(file_out):
            save_nc(dictionary=db.BT, file_out=file_out)
            print(f'\t{name_file}: OK')
        print('-'*85)
