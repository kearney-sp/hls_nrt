import os
import re

# input directory
inPATH_wc = '../data/extractions/tmp/co_wss*'

# input file for biomass ground data
inPATH_dat = '../data/ground/all_merged_infested_counties.csv'

# unique ID column name
id_col = 'Unique_ID'
# temporary ID column for preprocessing
id_col_tmp = None

# date column name
date_col = 'date'

# dependent variable column
y_col = 'Infested'

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
preprocess = None