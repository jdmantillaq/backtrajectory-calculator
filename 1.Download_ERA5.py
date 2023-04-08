# %%
import numpy as np
import pandas as pd
import cdsapi
import os
from BT import create_download_path

# define a dictionary with variable names and their corresponding
# parameter values
variables = {'uwnd': 'u_component_of_wind',
             'vwnd': 'v_component_of_wind',
             'sh': 'specific_humidity',
             'omega': 'vertical_velocity'}

# define the pressure level to be downloaded
level = [825]

# Define the directory path to save the downloaded data
directory = 'ERA5_data/'
# Create the download directory if it doesn't exist
create_download_path(directory)

# define the time range for which data needs to be downloaded
time_range = pd.date_range('2021-12-20', '2022-12-31', freq='D')

# define the geographic limits for downloading the information
lat_min = -15
lat_max = 25
lon_min = -110
lon_max = -20

# Define the time step (delta_t) for downloading the information
delta_t = 6

# Create a list of time steps formatted as strings
# (e.g., '00:00', '06:00', ...)
time_vector = [f'{i:02}:00' for i in list(range(0, 24, delta_t))]

# loop through all the variable names and their
# corresponding parameter values
for var_i, field_era in variables.items():
    print(var_i)
    # loop through all the pressure levels for which data needs
    # to be downloaded
    for level_i in level:
        # loop through all the dates in the time range for which
        # data needs to be downloaded
        for i, date in enumerate(time_range):
            # create a file name using variable name, pressure level,
            # date, and format
            file_i = f'{var_i}_{level_i}hPa_{date.strftime("%Y%m%d")}.nc'
            print(file_i)
            # create a path to save the downloaded data (one folder per year)
            path_i = f'{directory}{date.strftime("%Y")}/'

            # create the folder to store the downloaded data if it
            # does not already exist
            create_download_path(path_i)

            # check if the file already exists
            if not os.path.exists(f'{path_i}{file_i}'):
                print("\tDownloading...")
                # connect to the Climate Data Store API to retrieve data
                c = cdsapi.Client()
                # specify the parameters for the data download
                c.retrieve(
                    'reanalysis-era5-pressure-levels',
                    {
                        'product_type': 'reanalysis',
                        'format': 'netcdf',
                        'variable': field_era,
                        'pressure_level': f'{level_i}',
                        'year': f'{date.strftime("%Y")}',
                        'month': f'{date.strftime("%m")}',
                        'day': f'{date.strftime("%d")}',
                        'time': time_vector,
                        'area': [lat_max, lon_min, lat_min, lon_max],
                    },
                    # specify the path to save the downloaded file
                    f'{path_i}{file_i}')


# %%
