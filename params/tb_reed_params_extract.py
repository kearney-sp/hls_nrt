import os
import re
import pandas as pd
from hlsstack.hls_funcs import indices
import numpy as np
import datetime

# dask cluster location
cluster_loc = 'hpc'

# input path
inPATH = '../data/ground/Reed_Pastures.shp'

# set nickname to shorten path
nickname = 'tb_reed'

# output directory
outDIR = '../data/extractions/'

# output path basename
if nickname is not None:
    basename =  nickname + '_hls.csv'
else:
    basename =  re.sub('.csv', '_hls.csv', os.path.basename(inPATH))

# unique ID column name
id_col = 'filename'
# date column name
date_col = 'Date'

# coordinate reference system (CRS) in ESPG format of the input data
input_epsg = 26913

# coordinate reference system (CRS) in ESPG format of the output data
output_epsg = 26913

# specify buffer (in output_espg units) surrounding the point
# NOTE: a buffer of at least ~21.22 m around a point is required to 
# ensure a 30 m cell center is captured by the resulting polygon
buffer = -1

# dictionary specifying functions for each vegetation index to calculate and extract
veg_dict = {
    'NDVI': indices.ndvi_func,
    'DFI': indices.dfi_func,
    'NDTI': indices.ndti_func,
    'SATVI': indices.satvi_func,
    'NDII7': indices.ndii7_func,
    'SAVI': indices.savi_func,
    'RDVI': indices.rdvi_func,
    'MTVI1': indices.mtvi1_func,
    'NCI': indices.nci_func,
    'NDCI': indices.ndci_func,
    'PSRI': indices.psri_func,
    'NDWI': indices.ndwi_func,
    'EVI': indices.evi_func,
    'TCBI': indices.tcbi_func,
    'TCGI': indices.tcgi_func,
    'TCWI': indices.tcwi_func,
    'BAI_126': indices.bai_126_func,
    'BAI_136': indices.bai_136_func,
    'BAI_146': indices.bai_146_func,
    'BAI_236': indices.bai_236_func,
    'BAI_246': indices.bai_246_func,
    'BAI_346': indices.bai_346_func
}

# dictionary specifying individual bands to extract
band_list = ['BLUE', 'GREEN', 'RED', 'NIR1', 'SWIR1', 'SWIR2']

# define range of dates to extract data
yr_range = np.arange(2019, 2022)
mo_start = 1
day_start = 1
mo_end = 12
day_end = 31

# define function to preprocess data or set to None to skip
def preprocess(gdf):
    # dissolve by pasture name
    gdf = gdf.dissolve(id_col).reset_index()
    gdf_list = []
    for i in gdf['filename']:
        for yr in yr_range:
            gdf_list.append(pd.DataFrame({'filename': i,
                                          date_col: pd.date_range(datetime.date(yr, mo_start, day_start),
                                                                  datetime.date(yr, mo_end, day_end))}))
    gdf = pd.merge(gdf, pd.concat(gdf_list), on='filename', how='right')
    gdf = gdf.dropna(subset='Date')
    return gdf