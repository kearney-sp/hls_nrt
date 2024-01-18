import os
import re

# input directory
inPATH_wc = '../data/extractions/tmp/ltar_gb*'

# input file for biomass ground data
inPATH_dat = '../data/ground/herb-biomass-gb-ltar.csv'

# unique ID column name
id_col = 'id'

# date column name
date_col = 'date_median'

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

# whether to apply Bolton mask
mask_bolton_by_id = False

# custom preprocessing for Great Basin dataset
def preprocess(df):
    # create column of total herbaceous standing biomass
    df.loc[df['forb_kg_ha'].isnull(), 'forb_kg_ha'] = 0.0
    df.loc[df['grass_kg_ha'].isnull(), 'grass_kg_ha'] = 0.0
    df[y_col] = df['forb_kg_ha'] + df['grass_kg_ha']
    return df