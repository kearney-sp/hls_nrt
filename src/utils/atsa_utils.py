import numpy as np
import math
import statsmodels.formula.api as smf
import pandas as pd


def Compute_hot(data_blue, data_red, rmin, rmax, n_bin):
    bin_size = (rmax - rmin) / float(n_bin)
    x = np.zeros(n_bin, dtype=float)
    y = np.zeros(n_bin, dtype=float)
    ii = 0

    # find samples on clear line
    for i in range(0, n_bin):
        ind_bin = np.logical_and(data_blue >= rmin + i * bin_size, data_blue < rmin + (i + 1) * bin_size)
        num_bin = np.sum(ind_bin)
        if num_bin >= 20:
            x_bin = data_blue[ind_bin]
            y_bin = data_red[ind_bin]
            # remove outliers
            ind_good = y_bin <= np.mean(y_bin) + 3.0 * np.std(y_bin)
            num_good = np.sum(ind_good)
            x_bin = x_bin[ind_good]
            y_bin = y_bin[ind_good]
            order = np.argsort(y_bin)
            top_num = np.min([20, math.ceil(0.01 * num_good)])
            x_bin_select = x_bin[order[num_good - top_num:num_good]]
            y_bin_select = y_bin[order[num_good - top_num:num_good]]
            x[ii] = np.mean(x_bin_select)
            y[ii] = np.mean(y_bin_select)
            ii = ii + 1

    ind_0 = x > 0
    num_sample = np.sum(ind_0)
    x = x[ind_0]
    y = y[ind_0]
    x = pd.Series(x)
    y = pd.Series(y)
    fitdata = pd.DataFrame()
    fitdata['x'] = x
    fitdata['y'] = y
    if num_sample >= 0.5 * n_bin:
        # compute slope of clear line
        mod = smf.quantreg('y ~ x', fitdata)
        result = mod.fit(q=.5, max_iter=5000)

        result1 = result.params.x
        result0 = result.params.Intercept
        slop = result.params.x
        intercept = result.params.Intercept
        if result.params.x <= 1.5:
            result1 = 1.5
            result0 = np.mean(y) - result1 * np.mean(x)
            slop = 0
            intercept = 0
    else:
        result1 = 1.5
        result0 = np.mean(y) - 1.5 * np.mean(x)
        slop = 0
        intercept = 0

    hot_img = abs(data_blue * result1 - data_red + result0) / (1.0 + result1 ** 2) ** 0.5
    hot_img = np.ravel(hot_img)
    return hot_img, slop, intercept