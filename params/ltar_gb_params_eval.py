import os

nickname = 'ltar_gb'
kfold_group = 'year'
tuneby_group = 'year'

logo_group = 'kfold'
mod_col = 'Source'

inDIR = '../data/modeling/tmp'

drop_cols = ['Lat', 'long', 'forb_kg_ha', 'grass_kg_ha', 'litter_kg_ha', 'season', 'low', 'high', 'shape', 'radius_m', 'Year']
id_cols = ['kfold', 'id', 'pasture', 'date_median', 'year', 'Observed']
if logo_group in id_cols:
    id_cols.remove(logo_group)

# unique sub-plot ID column name
id_col_sub = None
# unique plot ID column name
id_col = 'id'
# date column name
date_col = 'date_median'
# pasture column name
past_col = 'pasture'
# grouping columns (e.g., for use when multiple dates might exist when grouping plots to pastures)
group_cols = ['year']

plot_group_cols = [mod_col, id_col]

# dependent variable column
y_col = 'Biomass_kg_ha'

inPATH = os.path.join(inDIR, nickname + '_cv_' + kfold_group + '_tuneby_' + tuneby_group + '_tmp.csv')

var_names = [
    'NDVI', 'DFI', 'NDTI', 'SATVI', 'NDII7', 'SAVI',
    'RDVI', 'MTVI1', 'NCI', 'NDCI', 'PSRI', 'NDWI', 'EVI', 'TCBI', 'TCGI', 'TCWI',
    'BAI_126', 'BAI_136', 'BAI_146', 'BAI_236', 'BAI_246', 'BAI_346',
    'BLUE', 'GREEN', 'RED', 'NIR1', 'SWIR1', 'SWIR2'
]