import os

nickname = 'ltar_jer'
kfold_group = 'Year'
tuneby_group = 'Year'

logo_group = 'kfold'
mod_col = 'Source'

inDIR = '../data/modeling/tmp'

drop_cols = ['site', 'zone', 'year', 'sampling_start', 'samplng_end', 'quadrats',
             'A_FORB', 'A_GRASS', 'P_FORB', 'P_GRASS', 'P_LF-SU', 'P_S-SHR', 'P_SHRUB', 'P_ST-SU', 'UNK_FORB', 'UNK_GRASS', 'YUEL']
id_cols = ['kfold', 'UnitID', 'pasture', 'Date', 'Year', 'season', 'Observed']
if logo_group in id_cols:
    id_cols.remove(logo_group)

# unique sub-plot ID column name
id_col_sub = None
# unique plot ID column name
id_col = 'UnitID'
# date column name
date_col = 'Date'
# pasture column name
past_col = 'pasture'
# grouping columns (e.g., for use when multiple dates might exist when grouping plots to pastures)
group_cols = ['Year', 'season']

plot_group_cols = [mod_col, id_col]

# dependent variable column
y_col = 'herb_kg_ha'

inPATH = os.path.join(inDIR, nickname + '_cv_' + kfold_group + '_tuneby_' + tuneby_group + '_tmp.csv')

var_names = [
    'NDVI', 'DFI', 'NDTI', 'SATVI', 'NDII7', 'SAVI',
    'RDVI', 'MTVI1', 'NCI', 'NDCI', 'PSRI', 'NDWI', 'EVI', 'TCBI', 'TCGI', 'TCWI',
    'BAI_126', 'BAI_136', 'BAI_146', 'BAI_236', 'BAI_246', 'BAI_346',
    'BLUE', 'GREEN', 'RED', 'NIR1', 'SWIR1', 'SWIR2'
]