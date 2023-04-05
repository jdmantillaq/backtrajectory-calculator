# Lagrangian Backtrajectories with ERA5 Data

This repository contains Python code to calculate backtrajectories using a Lagrangian method based on ERA5 reanalysis data.

## Data Source
The data used in this code is from the ERA5 dataset, which is a global atmospheric reanalysis dataset provided by the European Centre for Medium-Range Weather Forecasts (ECMWF). ERA5 provides high-resolution, consistent and quality-controlled data for a wide range of atmospheric variables, such as wind components, specific humidity, and vertical velocity. The data is available from 1979 to present and has a spatial resolution of approximately 31 km on 137 pressure levels.


## Code Description

The script, `1.Download_ERA5.py`, downloads the required ERA5 data using the Climate Data Store API. The script also contains utility functions for walking through directories, creating new directory paths, and downloading the ERA5 data.

#### Variables:
-   `variables`: A dictionary with variable names and their corresponding parameter values.
-   `level`: The pressure level to be downloaded.
-   `directory`: The directory path to save the downloaded data.
-   `time_range`: The time range for which data needs to be downloaded.
-   `lat_min`, `lat_max`, `lon_min`, `lon_max`: The geographic limits for downloading the information.


## License
This project is licensed under the MIT License - see the `LICENSE` file for details.



