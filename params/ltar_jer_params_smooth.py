import os
import re
import pandas as pd

# input directory
inPATH_wc = '../data/extractions/tmp/ltar_jer*'

# input file for biomass ground data
inPATH_dat = '../data/ground/ltar_jer_biomass_fg.csv'

# unique ID column name
id_col = 'UnitID'
# temporary ID column for preprocessing
id_col_tmp = None

# date column name
date_col = 'Date'

# dependent variable column
y_col = 'herb_kg_ha'

# output directory
outDIR = '../data/extractions'

# output file
outPATH = os.path.join(outDIR, re.sub('.csv', '_hls.csv', os.path.basename(inPATH_dat)))

# vegetation indices
veg_list = ['NDVI', 'DFI', 'NDTI', 'SATVI', 'NDII7', 
            'SAVI', 'RDVI', 'MTVI1', 'NCI', 'NDCI', 'PSRI', 'NDWI', 'EVI', 'TCBI', 'TCGI', 'TCWI',
            'BAI_126', 'BAI_136', 'BAI_146', 'BAI_236', 'BAI_246', 'BAI_346']

# individual bands
band_list = ['BLUE', 'GREEN', 'RED', 'NIR1', 'SWIR1', 'SWIR2']


past_dict = {
    'CDRRC Pasture 3': 'CP3N',
    'JER Pasture 9': 'JP9',
    'JER Pasture 12A': 'JP12A'
}

# whether to apply Bolton mask
mask_bolton_by_id = False

# define function to preprocess data or set to None to skip
def preprocess(df):
    df['sampling_start'] = pd.to_datetime(df['sampling_start'])
    df['sampling_end'] = pd.to_datetime(df['sampling_end'])
    df['pasture'] = df['site'].transform(lambda x: past_dict[x])
    df['UnitID'] = df.apply(lambda x: x['pasture'] + x['zone'], axis=1)
    df['herb_kg_ha'] = df[['A_FORB', 'A_GRASS', 'P_FORB', 'P_GRASS', 'UNK_FORB', 'UNK_GRASS']].sum(axis=1) * 10
    df[date_col] = df['sampling_start'] + (df['sampling_end'] - df['sampling_start']) / 2
    df = df[df[date_col].dt.year >= 2013]
    return df