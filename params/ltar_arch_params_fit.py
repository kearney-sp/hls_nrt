import os
import pandas as pd
import numpy as np

# NOTES:
# seems to be one very severe outlier of > 40,000 kg/ha. To be removed in preprocessing

# dask cluster location
cluster_loc = 'hpc'
tuneby_group = 'Pasture'
kfold_group = 'Pasture'

kfold_type = 'group_k'
tune_kfold_type = 'group_k'

use_cuda = False
# set backend as one of 'multiprocessing' or 'dask'
backend = 'dask' 

inDIR = '../data/extractions/'
inFILE = 'ltar_arch_biomass_hls.csv'
nickname = 'ltar_arch'

inPATH = os.path.join(inDIR, inFILE)

outDIR = '../data/modeling/'

outFILE_tmp = os.path.join(outDIR, 'tmp', nickname + '_cv_' + kfold_group + '_tuneby_' + tuneby_group + '_tmp.csv')

# unique ID column name
id_col = 'Pasture'
# date column name
date_col = 'Date_mean'
# dependent variable column
y_col = 'Biomass_kg_ha'

# apply transformation to dependent variable
y_col_xfrm = False
def y_col_xfrm_func(x):
    return(x*10)

# apply transformation to the output of the CPER (2022) model
cper_mod_xfrm = False
def cper_mod_xfrm_func(x):
    # convert from kg/ha to lbs/acre
    return(x * 0.892179122)

var_names = [
    'NDVI', 'DFI', 'NDTI', 'SATVI', 'NDII7', 'SAVI',
    'RDVI', 'MTVI1', 'NCI', 'NDCI', 'PSRI', 'NDWI', 'EVI', 'TCBI', 'TCGI', 'TCWI',
    'BAI_126', 'BAI_136', 'BAI_146', 'BAI_236', 'BAI_246', 'BAI_346',
    'BLUE', 'GREEN', 'RED', 'NIR1', 'SWIR1', 'SWIR2'
]

def load_df(inPATH, date_col, group_pastures=True):
    # Preprocessing steps here
    df = pd.read_csv(inPATH, parse_dates=[date_col])
    
    df['Year'] = df[date_col].dt.year

    # set 0.0 values for biomass to be NaN (based on email conversation)
    df.loc[df[y_col] == 0.0, y_col] = np.nan
    
    # check for any missing data
    missing_len = len(df[df[var_names].isnull().any(axis=1)])
    if missing_len > 0:
        print('Number of observations missing HLS data: ', str(missing_len))
        # remove missing data
        print('Removing observations missing data.')
        df = df[~df[var_names].isnull().any(axis=1)].copy()

    missing_y = len(df[df[y_col].isnull()])
    if missing_len > 0:
        print('Number of observations missing dependent variable data: ', str(missing_y))
        # remove missing data
        print('Removing observations missing data.')
        df = df[~df[y_col].isnull()].copy()

    # remove suspected outlier
    df = df[df[y_col] < 20000]
    print('Removing outlier with dependent variable greater than 20,000 kg/ha')

    if group_pastures:
        # group by pasture before fitting models
        print('Grouping data by pasture')
        df = df.groupby(['Pasture', date_col, 'Treatment', 'Year']).mean().reset_index()

    return df