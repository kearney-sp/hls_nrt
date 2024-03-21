import os

nickname = 'ffar_ok'
kfold_group = 'Block_Id'
tuneby_group = 'Block_Id'

logo_group = 'kfold'
mod_col = 'Source'

inDIR = '../data/modeling/tmp'

#drop_cols = ['Date', 'lat', 'long', 'Low', 'High', 'PP_g', 'ID_yr']
#anything that is not an ID column that we are not interested in, ie VOR readings etc. 
drop_cols = ['lat', 'long', 'Low']
id_cols = ['kfold', 'ID', 'Ranch', 'Block_Id', 'Treatment', 'Date', 'Year']
id_cols = id_cols + ['Observed']
if logo_group in id_cols:
    id_cols.remove(logo_group)

# unique sub-plot ID column name
id_col_sub = None
# unique plot ID column name
id_col = 'ID'
# date column name
date_col = 'Date'
# pasture column name
past_col = 'Block_Id'
# grouping columns (e.g., for use when multiple dates might exist when grouping plots to pastures)
group_cols = ['Year', 'Block_Id', 'Treatment', 'Ranch']

plot_group_cols = [mod_col, id_col]

# dependent variable column
y_col = 'mean_biomass'

inPATH = os.path.join(inDIR, nickname + '_cv_' + kfold_group + '_tuneby_' + tuneby_group + '_tmp.csv')

var_names = [
    'NDVI', 'DFI', 'NDTI', 'SATVI', 'NDII7', 'SAVI',
    'RDVI', 'MTVI1', 'NCI', 'NDCI', 'PSRI', 'NDWI', 'EVI', 'TCBI', 'TCGI', 'TCWI',
    'BAI_126', 'BAI_136', 'BAI_146', 'BAI_236', 'BAI_246', 'BAI_346',
    'BLUE', 'GREEN', 'RED', 'NIR1', 'SWIR1', 'SWIR2'
]