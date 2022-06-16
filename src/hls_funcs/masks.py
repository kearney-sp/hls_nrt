import dask
import numpy as np
import xarray as xr
from rasterio import features


def mask_hls(da, mask_types=['all']):
    band_QA = da.astype('int16')

    def unpack_bits(b_num):
        mask = np.subtract(np.divide(band_QA, 2 ** b_num).astype('int'),
                           (np.divide(np.divide(band_QA, 2 ** b_num).astype('int'), 2).astype('int') * 2))
        return mask

    def mask_from_QA(dat, bits):
        return xr.apply_ufunc(unpack_bits, bits).astype('int8')

    cirrus_mask = mask_from_QA(band_QA, 0)
    cloud_mask = mask_from_QA(band_QA, 1)
    cloud_adj_mask = mask_from_QA(band_QA, 2)
    shadow_mask = mask_from_QA(band_QA, 3)
    snow_mask = mask_from_QA(band_QA, 4)
    water_mask = mask_from_QA(band_QA, 5)
    low_aerosol_mask = np.logical_and(mask_from_QA(band_QA, 6) == 0, mask_from_QA(band_QA, 7) == 1).astype('int8')
    mod_aerosol_mask = np.logical_and(mask_from_QA(band_QA, 6) == 1, mask_from_QA(band_QA, 7) == 0).astype('int8')
    hi_aerosol_mask = np.logical_and(mask_from_QA(band_QA, 6) == 1, mask_from_QA(band_QA, 7) == 1).astype('int8')
    all_aerosol_mask = xr.concat([low_aerosol_mask, mod_aerosol_mask, hi_aerosol_mask], 
                                 dim='band').max(dim='band')
    
    mask_dict = {
        'cirrus': cirrus_mask,
        'cloud': cloud_mask,
        'cloud_adj': cloud_adj_mask,
        'shadow': shadow_mask,
        'snow': snow_mask,
        'water': water_mask,
        'low_aerosol': low_aerosol_mask,
        'mod_aerosol': mod_aerosol_mask,
        'high_aerosol': hi_aerosol_mask,
        'any_aerosol': all_aerosol_mask
    }
    
    if 'all' in mask_types:
        all_masks = xr.concat([cirrus_mask, cloud_mask, cloud_adj_mask,
                               shadow_mask, snow_mask, water_mask, hi_aerosol_mask],
                              dim='band')
        QA_mask_all = all_masks.max(dim='band')
        return QA_mask_all
    elif len(mask_types) > 1:
        all_masks = xr.concat([mask_dict[x] for x in mask_types],
                      dim='band')
        QA_mask_all = all_masks.max(dim='band')
        return QA_mask_all
    elif len(mask_types) == 1:
        return mask_dict[mask_types[0]]
    else:
        print('ERROR in "mask_types" definition. mask_types must be a list and one of, "all", "cirrus", "cloud", "cloud_adj", "snow", "water".')

def bolton_mask(ds, time_dim='time'):
    from src.hls_funcs.bands import blue_func, swir2_func
    dat_blue = blue_func(ds)
    dat_swir2 = swir2_func(ds)

    def cloud_outlier_mask(da_blue):
        blue_ts = da_blue / 10000.0
        cloud_mask = np.zeros_like(blue_ts)
        for idx in range(len(blue_ts)):
            if not np.isnan(blue_ts[idx]):
                idx_clear = np.where(~np.isnan(blue_ts))[0]
                if idx == np.min(idx_clear):
                    continue
                else:
                    idx_pre = np.max(idx_clear[idx_clear < idx])
                    blue_diff = blue_ts[idx] - blue_ts[idx_pre]
                    cloud_thresh = 0.03 * (1 + (idx - idx_pre) / 30)
                    if blue_diff > cloud_thresh:
                        blue_ts[idx] = np.nan
                        cloud_mask[idx] = 1
                    else:
                        continue
            else:
                continue
        return cloud_mask

    def cloud_outlier_mask_xr(dat, dims):
        xr_cm = xr.apply_ufunc(cloud_outlier_mask,
                               dat,
                               input_core_dims=[dims],
                               output_core_dims=[dims],
                               dask='parallelized', vectorize=True,
                               output_dtypes=[np.float])
        return xr_cm

    def shadow_outlier_mask(da_swir2):
        swir2_ts = da_swir2.copy()
        shadow_mask = np.zeros_like(swir2_ts)
        for idx in range(len(swir2_ts)):
            if not np.isnan(swir2_ts[idx]):
                idx_clear = np.where(~np.isnan(swir2_ts))[0]
                if idx == np.min(idx_clear):
                    continue
                elif idx == np.max(idx_clear):
                    try:
                        idx_pre = idx_clear[idx_clear < idx][-1]
                        idx_pre2 = idx_clear[idx_clear < idx][-2]
                        y = np.array([swir2_ts[idx_pre2], swir2_ts[idx_pre]])
                        x = np.array([idx_pre2, idx_pre])
                        dx = np.diff(x)
                        dy = np.diff(y)
                        slope = dy / dx
                        swir2_interp = swir2_ts[idx_pre] + slope[0] * (idx - idx_pre)
                        swir2_diff = swir2_interp - swir2_ts[idx]
                        shadow_val = swir2_diff / (swir2_ts[idx_pre] - swir2_ts[idx_pre2])
                        if (idx - idx_pre2 < 45) & (swir2_diff > 500) & (np.abs(shadow_val) > 2):
                            swir2_ts[idx] = np.nan
                            shadow_mask[idx] = 1
                        else:
                            continue
                    except IndexError:
                        continue
                else:
                    idx_pre = idx_clear[idx_clear < idx][-1]
                    idx_post = idx_clear[idx_clear > idx][0]
                    y = np.array([swir2_ts[idx_pre], swir2_ts[idx_post]])
                    x = np.array([idx_pre, idx_post])
                    dx = np.diff(x)
                    dy = np.diff(y)
                    slope = dy / dx
                    swir2_interp = swir2_ts[idx_pre] + slope[0] * (idx - idx_pre)
                    swir2_diff = swir2_interp - swir2_ts[idx]
                    shadow_val = swir2_diff / (swir2_ts[idx_post] - swir2_ts[idx_pre])
                    if (idx_post - idx_pre < 45) & (swir2_diff > 500) & (np.abs(shadow_val) > 2):
                        swir2_ts[idx] = np.nan
                        shadow_mask[idx] = 1
                    else:
                        continue
            else:
                continue
        return shadow_mask

    def shadow_outlier_mask_xr(dat, dims):
        xr_sm = xr.apply_ufunc(shadow_outlier_mask,
                               dat,
                               input_core_dims=[dims],
                               output_core_dims=[dims],
                               dask='parallelized',
                               vectorize=True,
                               output_dtypes=[np.float])
        return xr_sm

    shadow_outliers = shadow_outlier_mask_xr(dat_swir2, [time_dim]).transpose(time_dim, 'y', 'x')
    dat_blue = dat_blue.where(shadow_outliers == 0)
    cloud_outliers = cloud_outlier_mask_xr(dat_blue, [time_dim]).transpose(time_dim, 'y', 'x')
    mask = np.maximum(cloud_outliers, shadow_outliers)
    return mask


def shp2mask(shp, xr_object, transform, outshape, fill=0, dtype='int16', **kwargs):
    raster = features.rasterize(shp, fill=fill, transform=transform,
                                out_shape=outshape, dtype=dtype, **kwargs)
    return xr.DataArray(raster,
                        coords=(xr_object.coords['y'].values, xr_object.coords['x']),
                        dims=('y', 'x'))