import numpy as np
import xarray as xr
import pandas as pd
from scipy.signal import savgol_filter


def non_uniform_savgol(x, y, window, polynom):
    """
  Applies a Savitzky-Golay filter to y with non-uniform spacing
  as defined in x

  This is based on https://dsp.stackexchange.com/questions/1676/savitzky-golay-smoothing-filter-for-not-equally-spaced-data
  The borders are interpolated like scipy.signal.savgol_filter would do

  Parameters
  ----------
  x : array_like
      List of floats representing the x values of the data
  y : array_like
      List of floats representing the y values. Must have same length
      as x
  window : int (odd)
      Window length of datapoints. Must be odd and smaller than x
  polynom : int
      The order of polynom used. Must be smaller than the window size

  Returns
  -------
  np.array of float
      The smoothed y values
  """
    if len(x) != len(y):
        raise ValueError('"x" and "y" must be of the same size')

    if len(x) < window:
        raise ValueError('The data size must be larger than the window size')

    if type(window) is not int:
        raise TypeError('"window" must be an integer')

    if window % 2 == 0:
        raise ValueError('The "window" must be an odd integer')

    if type(polynom) is not int:
        raise TypeError('"polynom" must be an integer')

    if polynom >= window:
        raise ValueError('"polynom" must be less than "window"')

    half_window = window // 2
    polynom += 1

    # Initialize variables
    A = np.empty((window, polynom))  # Matrix
    tA = np.empty((polynom, window))  # Transposed matrix
    t = np.empty(window)  # Local x variables
    y_smoothed = np.full(len(y), np.nan)

    # Start smoothing
    for i in range(half_window, len(x) - half_window, 1):
        # Center a window of x values on x[i]
        for j in range(0, window, 1):
            t[j] = x[i + j - half_window] - x[i]

        # Create the initial matrix A and its transposed form tA
        for j in range(0, window, 1):
            r = 1.0
            for k in range(0, polynom, 1):
                A[j, k] = r
                tA[k, j] = r
                r *= t[j]

        # Multiply the two matrices
        tAA = np.matmul(tA, A)

        # Invert the product of the matrices
        tAA = np.linalg.inv(tAA)

        # Calculate the pseudoinverse of the design matrix
        coeffs = np.matmul(tAA, tA)

        # Calculate c0 which is also the y value for y[i]
        y_smoothed[i] = 0
        for j in range(0, window, 1):
            y_smoothed[i] += coeffs[0, j] * y[i + j - half_window]

        # If at the end or beginning, store all coefficients for the polynom
        if i == half_window:
            first_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    first_coeffs[k] += coeffs[k, j] * y[j]
        elif i == len(x) - half_window - 1:
            last_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    last_coeffs[k] += coeffs[k, j] * y[len(y) - window + j]

    # Interpolate the result at the left border
    for i in range(0, half_window, 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += first_coeffs[j] * x_i
            x_i *= x[i] - x[half_window]

    # Interpolate the result at the right border
    for i in range(len(x) - half_window, len(x), 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += last_coeffs[j] * x_i
            x_i *= x[i] - x[-half_window - 1]

    return y_smoothed


def double_savgol(ts, double=True, window1_min=11, window2=59, polynom1=3, polynom2=3, limit=61):
    ts_tmp = ts.copy()
    window1 = int(np.max([7, int(len(ts_tmp[~np.isnan(ts_tmp)]) / 4) // 2 * 2 + 1]))
    if double and len(ts_tmp[~np.isnan(ts_tmp)]) > window1_min:
        ts_tmp[~np.isnan(ts_tmp)] = non_uniform_savgol(np.arange(len(ts_tmp))[~np.isnan(ts_tmp)],
                                                       ts_tmp[~np.isnan(ts_tmp)],
                                                       window=window1, polynom=polynom1)
        ts_interp = pd.Series(ts_tmp)
        ts_interp = ts_interp.interpolate(method='linear', limit_area='inside', limit=limit)
        ts_interp = ts_interp.interpolate(method='linear', limit=None, limit_direction='both',
                                          limit_area='outside')
    else:
        ts_interp = pd.Series(ts_tmp)
        ts_interp = ts_interp.interpolate(method='linear', limit_area='inside', limit=limit)
        ts_interp = ts_interp.interpolate(method='linear', limit=None, limit_direction='both',
                                          limit_area='outside')
    try:
        ts_smooth = savgol_filter(ts_interp, window_length=window2, polyorder=polynom2)
    except np.linalg.LinAlgError:
        ts_smooth = ts_interp
    return ts_smooth


def modified_z_score(ts):
    # see https://towardsdatascience.com/removing-spikes-from-raman-spectra-8a9fdda0ac22
    median_int = np.nanmedian(ts)
    mad_int = np.nanmedian([np.abs(ts - median_int)])
    modified_z_scores = 0.6745 * (ts - median_int) / mad_int
    return modified_z_scores


def mask_ts_outliers(ts, threshold=3.5):
    # see https://towardsdatascience.com/removing-spikes-from-raman-spectra-8a9fdda0ac22
    ts_masked = ts.copy()
    ts_modz_robust = np.array(abs(modified_z_score(ts_masked)))
    if not np.all(np.isnan(ts_modz_robust)):
        spikes1 = ts_modz_robust > threshold
        ts_masked[spikes1] = np.nan
    return ts_masked


def despike_ts(dat_ts, dat_thresh, days_thresh, z_thresh=3.5, mask_outliers=False, iters=2):
    dat_ts_cln = dat_ts.copy()
    if mask_outliers:
        dat_ts_cln = mask_ts_outliers(dat_ts_cln, threshold=z_thresh)
    dat_mask = np.zeros_like(dat_ts_cln)
    for i in range(iters):
        for idx in range(len(dat_ts_cln)):
            if not np.isnan(dat_ts_cln[idx]):
                idx_clear = np.where(~np.isnan(dat_ts_cln))[0]
                if idx == np.min(idx_clear):
                    continue
                elif idx == np.max(idx_clear):
                    continue
                else:
                    idx_pre = idx_clear[idx_clear < idx][-1]
                    idx_post = idx_clear[idx_clear > idx][0]
                    y = np.array([dat_ts_cln[idx_pre], dat_ts_cln[idx_post]])
                    x = np.array([idx_pre, idx_post])
                    dx = np.diff(x)
                    dy = np.diff(y)
                    slope = dy / dx
                    dat_interp = dat_ts_cln[idx_pre] + slope[0] * (idx - idx_pre)
                    dat_diff = dat_interp - dat_ts_cln[idx]
                    shadow_val = dat_diff / (dat_ts_cln[idx_post] - dat_ts_cln[idx_pre])
                    if (idx_post - idx_pre < days_thresh) & (np.abs(dat_diff) > dat_thresh) & (np.abs(shadow_val) > 2):
                        dat_ts_cln[idx] = np.nan
                        dat_mask[idx] = 1
                    else:
                        continue
            else:
                continue
    dat_ts_cln[np.where(dat_mask == 1)] = np.nan
    return dat_ts_cln


def smooth_xr(dat, dims, kwargs={'double': True}):
    xr_smoothed = xr.apply_ufunc(double_savgol,
                                 dat,
                                 kwargs=kwargs,
                                 input_core_dims=[dims],
                                 output_core_dims=[dims],
                                 dask='parallelized', vectorize=True,
                                 output_dtypes=[float])
    return xr_smoothed.transpose('time', 'y', 'x')


def despike_ts_xr(dat, dat_thresh, dims, days_thresh=60, z_thresh=3.5, mask_outliers=False, iters=2):
    xr_ds = xr.apply_ufunc(despike_ts,
                           dat,
                           kwargs=dict(dat_thresh=dat_thresh,
                                       days_thresh=days_thresh,
                                       z_thresh=z_thresh,
                                       mask_outliers=mask_outliers,
                                       iters=iters),
                           input_core_dims=[dims],
                           output_core_dims=[dims],
                           dask='parallelized', vectorize=True,
                           output_dtypes=[float])
    return xr_ds.transpose('time', 'y', 'x')

