# %%

import numpy as np  # Mathematical operations and matrix handling
import matplotlib.pyplot as plt  # Graphics
import pandas as pd  # Date handling
import os
import glob
import cartopy.crs as ccrs
from BT import *


'''
Once all the desired backtrajectories have been calculated,
they can be processed.

This code will help you read all the generated information.
'''

# Path where the calculated BT files are stored for the specific model
path_files = 'processed_bt/'

archivos = np.sort(glob.glob(os.path.join(path_files, 'BT*.nc')))

# %%

files_date = pd.to_datetime([i.split('/')[-1].split('.')[-2]
                             for i in archivos], format='%Y%m%d')

# %%

# Clustering will be done only with 10 days (10*4)
num_days = 10
num_bt_dia = 4   # Number of BTs per day
dim_bt = num_days * num_bt_dia  # Time dimension of each BT

# Initialize arrays for latitude, longitude, specific humidity, and dates
lat = np.zeros((len(archivos), num_bt_dia, dim_bt)) * np.nan
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


# Starting coordinates for the calculation
lati = 6.25184
loni = -75.56359

#%%
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
    ax.plot(lon[di, :], lat[di, :], color='black', alpha=0.3, zorder=0)
    
for di in range(dates_dim):
    c = ax.scatter(lon[di, :], lat[di, :], c=sh[di, :], cmap=cmap,
                   vmin=vmin, vmax=vmax)
    
plt.colorbar(c, label='kg kg**-1')

ax.scatter(loni, lati, marker='*', c='firebrick', s=80)

# %%
