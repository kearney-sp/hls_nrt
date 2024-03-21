import os
import re

# input directory
inPATH_wc = '../data/extractions/tmp/tb_reed*'

# input file for biomass ground data
inPATH_dat = None

# unique ID column name
id_col = 'filename'

# date column name
date_col = 'Date'

# dependent variable column
y_col = None

# output directory
outDIR = '../data/extractions'

# output file
outPATH = os.path.join(outDIR, 'tb_reed_hls_ndvi.csv')

# vegetation indices
veg_list = ['NDVI']

# individual bands
band_list = []

# whether to apply Bolton mask
mask_bolton_by_id = False

# define function to preprocess data or set to None to skip
preprocess = None