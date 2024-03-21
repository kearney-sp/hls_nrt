import os
import re

# input directory
inPATH_wc = '../data/extractions/tmp/ffar_mi*'

# input file for biomass ground data
inPATH_dat = '../data/ground/ffar-mi-biomass.csv'

# unique ID column name
id_col = 'ID'
# temporary ID column for preprocessing
id_col_tmp = None

# date column name
date_col = 'Date'

# dependent variable column
y_col = 'mean_biomass'

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