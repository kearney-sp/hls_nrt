import os
import re
import pandas as pd
import geopandas as gdp
from hlsstack.hls_funcs import indices
import numpy as np
import datetime

# dask cluster location
cluster_loc = 'hpc'

# input path
inPATH = '../data/ground/JER_ecotone_quad_groups.shp'
inPATH_df = '../data/ground/ltar_jer_biomass_fg.csv'

# set nickname to shorten path
nickname = 'ltar_jer'

# output directory
outDIR = '../data/extractions/'

# output path basename
if nickname is not None:
    basename =  nickname + '_hls.csv'
else:
    basename =  re.sub('.csv', '_hls.csv', os.path.basename(inPATH))

# unique ID column name
id_col = 'UnitID'
# date column name
date_col = 'Date'

# coordinate reference system (CRS) in ESPG format of the input data
input_epsg = 4326

# coordinate reference system (CRS) in ESPG format of the output data
output_epsg = 32613

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

past_dict = {
    'CDRRC Pasture 3': 'CP3N',
    'JER Pasture 9': 'JP9',
    'JER Pasture 12A': 'JP12A'
}

df = pd.read_csv(inPATH_df, parse_dates=['sampling_start', 'sampling_end'])
df['pasture'] = df['site'].transform(lambda x: past_dict[x])
df['UnitID'] = df.apply(lambda x: x['pasture'] + x['zone'], axis=1)
df['herb_kg_ha'] = df[['A_FORB', 'A_GRASS', 'P_FORB', 'P_GRASS', 'UNK_FORB', 'UNK_GRASS']].sum(axis=1) * 10
df['Date'] = df['sampling_start'] + (df['sampling_end'] - df['sampling_start']) / 2
df = df[df['Date'].dt.year >= 2013]

# define function to preprocess data or set to None to skip
def preprocess(gdf, df=df):
    gdf = gdp.GeoDataFrame(pd.merge(df, gdf, on='UnitID', how='left'), geometry='geometry')
    # dissolve by pasture name
    #gdf = gdf.dissolve(id_col).reset_index()
    gdf = gdf.dropna(subset='Date')
    return gdf