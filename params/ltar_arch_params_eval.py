import os

nickname = 'ltar_arch'
kfold_group = 'Pasture'
tuneby_group = 'Pasture'

logo_group = 'kfold'
mod_col = 'Source'

inDIR = '../data/modeling/tmp'

#drop_cols = ['Date', 'lat', 'long', 'Low', 'High', 'PP_g', 'ID_yr']
drop_cols = ['lat', 'long', 'Low']
id_cols = ['kfold', 'ID', 'Pasture', 'Treatment', 'Date_mean', 'Year', 'Season', 'Observed']
if logo_group in id_cols:
    id_cols.remove(logo_group)

# unique sub-plot ID column name
id_col_sub = None
# unique plot ID column name
id_col = 'Pasture'
# date column name
date_col = 'Date_mean'
# pasture column name
past_col = 'Pasture'
# grouping columns (e.g., for use when multiple dates might exist when grouping plots to pastures)
group_cols = ['Year', 'Treatment', 'Season']

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