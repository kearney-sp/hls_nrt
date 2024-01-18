import os
import re
import pandas as pd

# input directory
inPATH_wc = '../data/extractions/tmp/ltar_reno*'

# input file for biomass ground data
inPATH_dat = '../data/ground/ltar_reno_biomass.csv'

# unique ID column name
id_col = 'Pasture'
# temporary ID column for preprocessing
id_col_tmp = None

# date column name
date_col = 'Date'

# dependent variable column
y_col = 'DW kg/ha'

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
    # get mean of replicates by pasture and date
    df = df.groupby([id_col, date_col]).mean().reset_index()
    return df