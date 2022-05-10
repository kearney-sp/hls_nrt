import requests as r
import os
from netrc import netrc
from subprocess import Popen
import stackstac
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import certifi
from pyproj import Transformer


# Create a dictionary (i.e., look-up table; LUT) including the HLS product bands mapped to names
lut = {'HLSS30':
       {'B01': 'COASTAL-AEROSOL',
        'B02': 'BLUE', 
        'B03': 'GREEN', 
        'B04': 'RED', 
        'B05': 'RED-EDGE1',
        'B06': 'RED-EDGE2', 
        'B07': 'RED-EDGE3',
        'B08': 'NIR-Broad',
        'B8A': 'NIR1', 
        'B09': 'WATER-VAPOR',
        'B10': 'CIRRUS',
        'B11': 'SWIR1', 
        'B12': 'SWIR2', 
        'Fmask': 'FMASK'},
       'HLSL30': 
       {'B01': 'COASTAL-AEROSOL',
        'B02': 'BLUE', 
        'B03': 'GREEN', 
        'B04': 'RED', 
        'B05': 'NIR1',
        'B06': 'SWIR1',
        'B07': 'SWIR2', 
        'B09': 'CIRRUS', 
        'B10': 'TIR1', 
        'B11': 'TIR2', 
        'Fmask': 'FMASK'}}

# List of all available/acceptable band names
all_bands = ['ALL', 'COASTAL-AEROSOL', 'BLUE', 'GREEN', 'RED', 'RED-EDGE1', 'RED-EDGE2', 'RED-EDGE3', 
             'NIR1', 'SWIR1', 'SWIR2', 'CIRRUS', 'TIR1', 'TIR2', 'WATER-VAPOR', 'FMASK']

# list of just the bands currently used in functions
needed_bands = ['BLUE', 'GREEN', 'RED', 'NIR1', 'SWIR1', 'SWIR2', 'FMASK']


def HLS_CMR_STAC(hls_data, bbox_latlon, lim=100, aws=False):
    stac = 'https://cmr.earthdata.nasa.gov/stac/' # CMR-STAC API Endpoint
    stac_response = r.get(stac).json()            # Call the STAC API endpoint
    stac_lp = [s for s in stac_response['links'] if 'LP' in s['title']]  # Search for only LP-specific catalogs

    # LPCLOUD is the STAC catalog we will be using and exploring today
    lp_cloud = r.get([s for s in stac_lp if s['title'] == 'LPCLOUD'][0]['href']).json()
    lp_links = lp_cloud['links']
    lp_search = [l['href'] for l in lp_links if l['rel'] == 'search'][0]  # Define the search endpoint
    search_query = f"{lp_search}?&limit=100"    # Add in a limit parameter to retrieve 100 items at a time.
    bbox = f'{bbox_latlon[0]},{bbox_latlon[1]},{bbox_latlon[2]},{bbox_latlon[3]}'  # Defined from ROI bounds
    search_query2 = f"{search_query}&bbox={bbox}"                                                  # Add bbox to query
    date_time = hls_data['date_range'][0]+'/'+hls_data['date_range'][1]  # Define start time period / end time period
    search_query3 = f"{search_query2}&datetime={date_time}"  # Add to query that already includes bbox
    if lim > 100:
        s30_items = list()
        l30_items = list()
        for i in range(int(np.ceil(lim/100))):
            if i > 10:
                print('WARNING: Fetching more than 1000 records, this may result in a very large dataset.')
            collections = r.get(search_query3).json()['features']    
            hls_collections = [c for c in collections if 'HLS' in c['collection']]
            s30_items = s30_items + [h for h in hls_collections if h['collection'] == 'HLSS30.v2.0']  # Grab HLSS30 collection and append
            l30_items = l30_items + [h for h in hls_collections if h['collection'] == 'HLSL30.v2.0']  # Grab HLSL30 collection and append
            start_time = str(
                max(datetime.strptime(s30_items[-1]['properties']['datetime'].split('T')[0], '%Y-%m-%d'),
                    datetime.strptime(l30_items[-1]['properties']['datetime'].split('T')[0], '%Y-%m-%d')).date() + timedelta(days=1))
            date_time = start_time+'/'+hls_data['date_range'][1]
            search_query3 = f"{search_query2}&datetime={date_time}"  # update query with new start time 
            if len([h for h in hls_collections if h['collection'] == 'HLSS30.v2.0']) + len([h for h in hls_collections if h['collection'] == 'HLSL30.v2.0']) == 0:
                break # stop searching if no results are found
    else:
        collections = r.get(search_query3).json()['features']    
        hls_collections = [c for c in collections if 'HLS' in c['collection']]
        s30_items = [h for h in hls_collections if h['collection'] == 'HLSS30.v2.0']  # Grab HLSS30 collection
        l30_items = [h for h in hls_collections if h['collection'] == 'HLSL30.v2.0']  # Grab HLSL30 collection
    
    if aws:
        for stac in s30_items:
            for band in stac['assets']:
                stac['assets'][band]['href'] = stac['assets'][band]['href'].replace('https://lpdaac.earthdata.nasa.gov/lp-prod-protected', 
                                                                                    's3://lp-prod-protected')
                stac['assets'][band]['href'] = stac['assets'][band]['href'].replace('https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected', 
                                                                                    's3://lp-prod-protected')
                
        for stac in l30_items:
            for band in stac['assets']:
                stac['assets'][band]['href'] = stac['assets'][band]['href'].replace('https://lpdaac.earthdata.nasa.gov/lp-prod-protected', 
                                                                                    '/vsis3/lp-prod-protected')
                stac['assets'][band]['href'] = stac['assets'][band]['href'].replace('https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected', 
                                                                                    '/vsis3/lp-prod-protected')
    return {'S30': s30_items,
            'L30': l30_items}


def setup_netrc(creds, aws):
    urs = 'urs.earthdata.nasa.gov' 
    try:
        netrcDir = os.path.expanduser("~/.netrc")
        netrc(netrcDir).authenticators(urs)[0]
        del netrcDir

    # Below, create a netrc file and prompt user for NASA Earthdata Login Username and Password
    except FileNotFoundError:
        homeDir = os.path.expanduser("~")
        Popen('touch {0}.netrc | chmod og-rw {0}.netrc | echo machine {1} >> {0}.netrc'.format(homeDir + os.sep, urs), shell=True)
        Popen('echo login {} >> {}.netrc'.format(creds[0], homeDir + os.sep), shell=True)
        Popen('echo password {} >> {}.netrc'.format(creds[1], homeDir + os.sep), shell=True)
        del homeDir

    # Determine OS and edit netrc file if it exists but is not set up for NASA Earthdata Login
    except TypeError:
        homeDir = os.path.expanduser("~")
        Popen('echo machine {1} >> {0}.netrc'.format(homeDir + os.sep, urs), shell=True)
        Popen('echo login {} >> {}.netrc'.format(creds[0], homeDir + os.sep), shell=True)
        Popen('echo password {} >> {}.netrc'.format(creds[1], homeDir + os.sep), shell=True)
        del homeDir
    del urs
    if aws:
        return(r.get('https://lpdaac.earthdata.nasa.gov/s3credentials').json())
    else:
        return('')

    
def build_xr(stac_dict, lut=lut, bbox=None, stack_chunks=(4000, 4000), proj_epsg=32613):
    try:
        s30_stack = stackstac.stack(stac_dict['S30'], epsg=proj_epsg, resolution=30, assets=[i for i in lut['HLSS30'] if lut['HLSS30'][i] in needed_bands],
                                    bounds=bbox, chunksize=stack_chunks)
        s30_stack['band'] = [lut['HLSS30'][b] for b in s30_stack['band'].values]
        s30_stack['time'] = [datetime.fromtimestamp(t) for t in s30_stack.time.astype('int').values//1000000000]
        s30_stack = s30_stack.to_dataset(dim='band').reset_coords(['end_datetime', 'start_datetime'], drop=True)
    except ValueError:
        print('Warning: ValueError in S30 stacking.')
        s30_stack = None
    try:
        l30_stack = stackstac.stack(stac_dict['L30'], epsg=32613, resolution=30, assets=[i for i in lut['HLSL30'] if lut['HLSL30'][i] in needed_bands],
                                   bounds=bbox, chunksize=stack_chunks)
        l30_stack['band'] = [lut['HLSL30'][b] for b in l30_stack['band'].values]
        l30_stack['time'] = [datetime.fromtimestamp(t) for t in l30_stack.time.astype('int').values//1000000000]
        l30_stack = l30_stack.to_dataset(dim='band').reset_coords(['end_datetime', 'start_datetime'], drop=True)
    except ValueError:
        print('Warning: ValueError in L30 stacking.')
        l30_stack = None
    if s30_stack is not None and l30_stack is not None:
        hls_stack = xr.concat([s30_stack, l30_stack], dim='time')
    elif s30_stack is not None:
        hls_stack = s30_stack
    elif l30_stack is not None:
        hls_stack = l30_stack
    else:
        print('No data found for date range')
    return hls_stack.chunk({'time': 1, 'y': -1, 'x': -1})
    

def get_hls(hls_data={}, bbox=[517617.2187, 4514729.5, 527253.4091, 4524372.5], 
            lut=lut, lim=100, aws=False, stack_chunks=(4000, 4000), proj_epsg=32613):   
    # run functions
    transformer = Transformer.from_crs('epsg:' + str(proj_epsg), 'epsg:4326')
    bbox_lon, bbox_lat = transformer.transform(bbox[[0, 2]], bbox[[1,3]])
    bbox_latlon = list(np.array(list(map(list, zip(bbox_lat, bbox_lon)))).flatten())
    catalog = HLS_CMR_STAC(hls_data, bbox_latlon, lim, aws)
    da  = build_xr(catalog, lut, bbox, stack_chunks, proj_epsg)
    return da


def setup_env(aws=False, creds=[]):
    #define gdalenv
    if aws:
        # set up creds
        s3_cred = setup_netrc(creds, aws=aws)
        env = dict(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', 
                   #AWS_NO_SIGN_REQUEST='YES',
                   GDAL_MAX_RAW_BLOCK_CACHE_SIZE='200000000',
                   GDAL_SWATH_SIZE='200000000',
                   VSI_CURL_CACHE_SIZE='200000000',
                   CPL_VSIL_CURL_ALLOWED_EXTENSIONS='TIF',
                   GDAL_HTTP_UNSAFESSL='YES',
                   GDAL_HTTP_COOKIEFILE=os.path.expanduser('~/cookies.txt'),
                   GDAL_HTTP_COOKIEJAR=os.path.expanduser('~/cookies.txt'),
                   AWS_REGION='us-west-2',
                   AWS_SECRET_ACCESS_KEY=s3_cred['secretAccessKey'],
                   AWS_ACCESS_KEY_ID=s3_cred['accessKeyId'],
                   AWS_SESSION_TOKEN=s3_cred['sessionToken'],
                   CURL_CA_BUNDLE=certifi.where())
    else:
        env = dict(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', 
                   AWS_NO_SIGN_REQUEST='YES',
                   GDAL_MAX_RAW_BLOCK_CACHE_SIZE='200000000',
                   GDAL_SWATH_SIZE='200000000',
                   VSI_CURL_CACHE_SIZE='200000000',
                   GDAL_HTTP_COOKIEFILE=os.path.expanduser('~/cookies.txt'),
                   GDAL_HTTP_COOKIEJAR=os.path.expanduser('~/cookies.txt'),
                   CURL_CA_BUNDLE=certifi.where())

    os.environ.update(env)
    