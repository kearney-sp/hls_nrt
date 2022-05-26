import fiona
import geopandas as gpd
from zipfile import ZipFile
import os

def kmz_to_shp(kmz_path, tempDir):
    fiona.drvsupport.supported_drivers['libkml'] = 'rw' # enable KML support 
    fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'
    
    kmz = ZipFile(kmz_path, 'r')
    kmz.extract('doc.kml', tempDir)

    gdf = gpd.read_file(os.path.join(tempDir,'doc.kml'))
    
    return gdf
    