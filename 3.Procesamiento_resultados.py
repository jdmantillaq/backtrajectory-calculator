# %%

import numpy as np  # Operaciones matemáticas y manejo de matrices
import matplotlib.pyplot as plt  # Gráficos
import pandas as pd  # Manejo de fechas
import os
import glob
from netCDF4 import Dataset
import cartopy.crs as ccrs
import sys
from BT import *
sys.path.append('/home/jdmantillaq/Documents/Karel/')

import seaborn as sns
sns.set(style="whitegrid")
sns.set_context('notebook', font_scale=1.0)


def read_nc(file_i):
    import numpy as np  # Operaciones matemáticas y manejo de matrices
    import pandas as pd  # Manejo de fechas
    from netCDF4 import Dataset

    Variable = Dataset(file_i, 'r')

    dates = np.array(Variable.variables['time'][:])

    fechas = pd.to_datetime("1900-01-01 00:00:00") \
        + pd.to_timedelta(dates, unit='h')

    lon_values = np.array(Variable.variables['lon'][:])
    lat_values = np.array(Variable.variables['lat'][:])
    sh_values = np.array(Variable.variables['q'][:])
    plev_values = np.array(Variable.variables['level'][:])  # shape 4x40
    fechas = np.array(fechas).reshape(plev_values.shape)
    dates = dates.reshape(plev_values.shape)

    return fechas, dates, plev_values, lat_values, lon_values, sh_values

# -----------------------------------------------------------------------------
'''
Una vez ya se hayan calculadas todas las retrotrayectorias deseadas
estas se pueden procesar.


Este código te ayudará a leer toda la información generada.
'''


# ruta donde están los archivos de las BT calculadas para le modelo particular
path_files = f'/home/jdmantillaq/Documents/Karel/procesados/'

archivos = np.sort(glob.glob(os.path.join(path_files, 'BT*.nc')))

#%%

files_date = pd.to_datetime([i.split('/')[-1].split('.')[-2]
                             for i in archivos], format='%Y%m%d')

#%%



# se va a hacer el clusterin solo con 10 días (10*4)
num_days = 10
num_bt_dia = 4   # Cantidad de BTs, por día
dim_bt = num_days*num_bt_dia  # Dimensión temporal de cada BT


lat = np.zeros((len(archivos), num_bt_dia, dim_bt))*np.nan
lon = np.copy(lat)
sh = np.copy(lat)
dates = np.copy(lat)

for i, file_i in enumerate(archivos):
    try:
        fechas, dates_values, plev_values, lat_values, lon_values, sh_values =\
            read_nc(file_i)
    except:
        continue

    lat[i, :, :] = lat_values[:, :dim_bt]
    lon[i, :, :] = lon_values[:, :dim_bt]
    sh[i, :, :] = sh_values[:, :dim_bt]
    dates[i, :, :] = dates_values[:, :dim_bt]
    del(lat_values, lon_values, fechas, plev_values, dates_values)

print('Achivos OK')
lat = lat.reshape((len(archivos)*num_bt_dia, dim_bt))
lon = lon.reshape((len(archivos)*num_bt_dia, dim_bt))
sh = sh.reshape((len(archivos)*num_bt_dia, dim_bt))
dates = dates.reshape((len(archivos)*num_bt_dia, dim_bt))

fechas = pd.to_datetime("1900-01-01 00:00:00") \
                    + pd.to_timedelta(dates[:, 0], unit='h')


#%%
# Coordenadas de inicio del cálculo
lati = 4.60971
loni = -74.08175

figsize = (10, 6)
img_extent = (-95, -30, -20, 30)
cmap = 'viridis'
(vmin, vmax) = (0, 0.012)

fig = plt.figure(figsize=(figsize))
proj = ccrs.PlateCarree(central_longitude=0)
ax = fig.add_axes([0, 0, 1, 1], projection=proj)
ax = Continentes_lon_lat(ax)
ax.set_extent(img_extent, ccrs.PlateCarree())

(dates_dim, back_step_dim) = lon.shape
for di in range(dates_dim):
    c = ax.scatter(lon[di, :], lat[di, :], c=sh[di, :], cmap=cmap,
                   vmin=vmin, vmax=vmax)
plt.colorbar(c, label='kg kg**-1')

ax.scatter(loni, lati, marker='*', c='firebrick', s=80)
