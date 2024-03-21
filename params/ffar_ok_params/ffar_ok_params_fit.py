import os
import pandas as pd
import numpy as np

# NOTES:
# seems to be one very severe outlier of > 40,000 kg/ha. To be removed in preprocessing

# dask cluster location
cluster_loc = 'hpc'
tuneby_group = 'Block_Id'
kfold_group = 'Block_Id'

kfold_type = 'group_k'
tune_kfold_type = 'group_k'

use_cuda = False
# set backend as one of 'multiprocessing' or 'dask'
backend = 'dask' 

inDIR = '../data/extractions/'
inFILE = 'ffar-ok-biomass_hls.csv'
nickname = 'ffar_ok'

inPATH = os.path.join(inDIR, inFILE)

outDIR = '../data/modeling/'

outFILE_tmp = os.path.join(outDIR, 'tmp', nickname + '_cv_' + kfold_group + '_tuneby_' + tuneby_group + '_tmp.csv')

# unique ID column name
id_col = 'ID'
# date column name
date_col = 'Date'
# dependent variable column
y_col = 'mean_biomass'

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
    return df