# %%
import numpy as np
import pandas as pd
import cdsapi
import os


def walk_path_target(path, target):
    # function to walk through the directory path and find target file(s)
    import glob  # importing glob library to find pathnames matching a
    # specified pattern
    import os  # importing os library to interact with the operating system
    return np.sort([y for x in os.walk(path) for y
                    in glob.glob(os.path.join(x[0], target))])


def create_download_path(path):
    # function to create a new directory path
    os.makedirs(path, exist_ok=True)


# define a dictionary with variable names and their corresponding
# parameter values
variables = {'uwnd': 'u_component_of_wind',
             'vwnd': 'v_component_of_wind',
             'sh': 'specific_humidity',
             'omega': 'vertical_velocity'}

# define the pressure level to be downloaded
level = [825]

# define the directory path to save the downloaded data
directory = 'ERA5_data/'

# define the time range for which data needs to be downloaded
time_range = pd.date_range('1980-01-01', '1980-02-01', freq='D')

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
                        'time': ['00:00', '06:00', '12:00', '18:00'],
                        'area': [25, -100, -15, -34],
                    },
                    # specify the path to save the downloaded file
                    f'{path_i}{file_i}')




# %%
