import os
import re

# input directory
inPATH_wc = '../data/extractions/tmp/snaplands_rr*'

# input file for biomass ground data
inPATH_dat = '../data/ground/2023-2021_RobertsRanch-ResidualEstimates_USDA-ARS.csv'

# unique ID column name
id_col = 'ID'
# temporary ID column for preprocessing
id_col_tmp = 'SiteID'

# date column name
date_col = 'Date'

# dependent variable column
y_col = 'biomass_lb_ac'

# output directory
outDIR = '../data/extractions'

# output file
outPATH = os.path.join(outDIR, re.sub('.csv', '_hls_20231115.csv', os.path.basename(inPATH_dat)))

# vegetation indices
veg_list = ['NDVI', 'DFI', 'NDTI', 'SATVI', 'NDII7', 
            'SAVI', 'RDVI', 'MTVI1', 'NCI', 'NDCI', 'PSRI', 'NDWI', 'EVI', 'TCBI', 'TCGI', 'TCWI',
            'BAI_126', 'BAI_136', 'BAI_146', 'BAI_236', 'BAI_246', 'BAI_346']

# individual bands
band_list = ['BLUE', 'GREEN', 'RED', 'NIR1', 'SWIR1', 'SWIR2']

# whether to apply Bolton mask
mask_bolton_by_id = False

# custom preprocessing for Roberts Ranch dataset
def preprocess(df):
    # create unique ID based on date to address multiple polygons per plot in data
    df['ID'] = df.apply(lambda x: x[id_col_tmp] + '_' + str(x[date_col].date()), axis=1)
    return df