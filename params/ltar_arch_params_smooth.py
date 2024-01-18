import os
import re

# input directory
inPATH_wc = '../data/extractions/tmp/ltar_arch*'

# input file for biomass ground data
inPATH_dat = '../data/ground/ltar_arch_biomass.csv'

# unique ID column name
id_col = 'ID_yr'
# temporary ID column for preprocessing
id_col_tmp = 'ID'

# date column name
date_col = 'Date_mean'

# dependent variable column
y_col = 'Biomass_kg_ha'

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

# define function to preprocess data or set to None to skip
def preprocess(df):
    # create new ID column that is a combo of year and ID
    df[id_col] = df.apply(lambda x: str(x[id_col_tmp]) + '_' + str(x[date_col].year), axis=1)
    return df