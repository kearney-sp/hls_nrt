import os

nickname = 'rr_snaplands'
logo_group = 'Pasture'
tuneby = 'pasture'
mod_col = 'Source'

inDIR = '../data/modeling/tmp'

drop_cols = ['Lat', 'Long', '_geometry', 'biomass_kg_ha', 'sitePhotos']
id_cols = ['ID', 'SiteID', 'Pasture', 'Date', 'Year', 'Season', 'Observed']
id_cols.remove(logo_group)

# unique sub-plot ID column name
id_col_sub = None
# unique plot ID column name
id_col = 'SiteID'
# date column name
date_col = 'Date'
# pasture column name
past_col = 'Pasture'
# grouping columns (e.g., for use when multiple dates might exist when grouping plots to pastures)
group_cols = ['Year', 'Season']

plot_group_cols = [mod_col, id_col]

# dependent variable column
y_col = 'biomass_lb_ac'

inPATH = os.path.join(inDIR, nickname + '_cv_' + logo_group + '_tuneby_' + tuneby + '_tmp.csv')

var_names = [
    'NDVI', 'DFI', 'NDTI', 'SATVI', 'NDII7', 'SAVI',
    'RDVI', 'MTVI1', 'NCI', 'NDCI', 'PSRI', 'NDWI', 'EVI', 'TCBI', 'TCGI', 'TCWI',
    'BAI_126', 'BAI_136', 'BAI_146', 'BAI_236', 'BAI_246', 'BAI_346',
    'BLUE', 'GREEN', 'RED', 'NIR1', 'SWIR1', 'SWIR2'
]