import os
import re
from hlsstack.hls_funcs import indices

# NOTES:
# long and lat were mixed up in the dataset from ARCH - fixed manually in CSV

#### shared parameters for all steps ####
# set nickname to shorten path
nickname = 'ltar_arch'

# unique ID column name
id_col = 'ID'

# modified ID column for extracting - necessary when multiple locations exist with same ID
id_col_mod = 'ID_yr'

# date column name
date_col = 'Date_mean'

#### parameters for extraction of HLS data ####

# input path
inPATH = '../data/ground/ltar_arch_biomass.csv'

# output directory
outDIR = '../data/extractions/'

# create output path basename
if nickname is not None:
    basename =  nickname + '_hls.csv'
else:
    basename =  re.sub('.csv', '_hls.csv', os.path.basename(inPATH))


# coordinate column names
x_coord_col = 'long'
y_coord_col = 'lat'

# coordinate reference system (CRS) in ESPG format of the input data
input_epsg = 4326

# coordinate reference system (CRS) in ESPG format of the output data
output_epsg = 32617

# specify buffer (in output_espg units) surrounding the point
# NOTE: a buffer of at least ~21.22 m around a point is required to 
# ensure a 30 m cell center is captured by the resulting polygon
buffer = 22

# dictionary specifying functions for each vegetation index to calculate and extract
veg_dict = {
    'NDVI': indices.ndvi_func,
    'DFI': indices.dfi_func,
    'NDTI': indices.ndti_func,
    'SATVI': indices.satvi_func,
    'NDII7': indices.ndii7_func,
    'SAVI': indices.savi_func,
    'RDVI': indices.rdvi_func,
    'MTVI1': indices.mtvi1_func,
    'NCI': indices.nci_func,
    'NDCI': indices.ndci_func,
    'PSRI': indices.psri_func,
    'NDWI': indices.ndwi_func,
    'EVI': indices.evi_func,
    'TCBI': indices.tcbi_func,
    'TCGI': indices.tcgi_func,
    'TCWI': indices.tcwi_func,
    'BAI_126': indices.bai_126_func,
    'BAI_136': indices.bai_136_func,
    'BAI_146': indices.bai_146_func,
    'BAI_236': indices.bai_236_func,
    'BAI_246': indices.bai_246_func,
    'BAI_346': indices.bai_346_func
}

# dictionary specifying individual bands to extract
band_list = ['BLUE', 'GREEN', 'RED', 'NIR1', 'SWIR1', 'SWIR2']

# define function to preprocess data or set to None to skip
def preprocess(df):
    # create new ID column that is a combo of year and ID
    df[id_col_mod] = df.apply(lambda x: str(x[id_col]) + '_' + str(x[date_col].year), axis=1)
    return df