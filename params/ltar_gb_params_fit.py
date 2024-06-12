import os
import pandas as pd

# dask cluster location
cluster_loc = 'hpc'
tuneby_group = 'year'
kfold_group = 'year'

kfold_type = 'logo'
tune_kfold_type = 'logo'

use_cuda = False
# set backend as one of 'multiprocessing' or 'dask'
backend = 'dask' 

inDIR = '../data/extractions/'
inFILE = 'herb-biomass-gb-ltar_hls.csv'
nickname = 'ltar_gb'

inPATH = os.path.join(inDIR, inFILE)
inPATH2 = '../data/ground/LTVR_herbaceous_functional_type_biomass_2015-2022.csv'

outDIR = '../data/modeling/'

outFILE_tmp = os.path.join(outDIR, 'tmp', nickname + '_cv_' + kfold_group + '_tuneby_' + tuneby_group + '_tmp.csv')

# unique ID column name
id_col = 'id'
# date column name
date_col = 'date_median'
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

def load_df(inPATH, date_col):
    # Preprocessing steps here
    df = pd.read_csv(inPATH, parse_dates=[date_col])
    df_gb2 = pd.read_csv(inPATH2)
    df_gb2['site'] = df_gb2['vegtype'] + df_gb2['siteno'].astype('str')
    df_gb2 = df_gb2.groupby(['year', 'site', 'plot'])['g.m2'].sum().reset_index()
    df_gb2 = df_gb2.groupby(['year', 'site'])['g.m2'].mean().reset_index()
    df = pd.merge(df, 
                  df_gb2[['site', 'year', 'g.m2']], 
                  left_on=[id_col, 'year'], 
                  right_on=['site', 'year'], 
                  how='left')
    df = df.rename(columns={'herb_kg_ha': 'herb_kg_ha_old'})
    df['Biomass_kg_ha'] = df['g.m2'] * 10
    
    df['Year'] = df[date_col].dt.year

    # check for any missing data
    missing_len = len(df[df[var_names].isnull().any(axis=1)])
    print('Number of observations missing HLS data: ', str(missing_len))

    if missing_len > 0:
        # remove missing data
        print('Removing observations missing data.')
        df = df[~df[var_names].isnull().any(axis=1)].copy()

    return df