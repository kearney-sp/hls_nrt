# hls_nrt
Near-real-time applications for Harmonized Landsat-Sentinel data

## Objectives
The scripts in the main folder will update the CPER viewer app on gcloud. They are primarily designed to be run on SCINet (CERES). They pull near-real time Harmonized Landsat-Sentinel data from LPDAAC, smooth and gap-fill the data, apply saved models to compute weekly standing biomass, fractional vegetation cover and NDVI, compute means by pasture and then upload all maps and summary data to gcloud. The app will update automatically once new data are uploaded.

## Workflow
#### Download data to /90daydata
Run HLS2_aoi_yr_download.ipynb.  
Set the desired year and area of interest (AOI) using the *prefix* parameter. For now, really only works with 'cper' as the AOI prefix.
##### *Outputs*
Saves a single .nc file to disk in /90daydata/cper_neon_aop/--*prefix*--

#### Compute all the vegetation products
Run HLS2_aoi_veg_products.ipynb

#### Save all vegetation products in format for gcloud
Run HLS2_aoi_products_to_gcloud.ipynb

#### Compute and save all pasture means
Run HLS2_aoi_compute_means.ipynb

#### Upload everything to gcloud
Run HLS2_upload_to_gcloud.ipynb