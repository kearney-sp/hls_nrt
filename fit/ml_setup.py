import pandas as pd
import numpy as np
import dask
import pickle
import os
import time
import itertools
from joblib import parallel_backend
import statsmodels.formula.api as smf
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import Lasso, LinearRegression, ElasticNet
from sklearn.pipeline import Pipeline
import torch
import torch.nn as nn
import torch.optim as optim
import skorch
from skorch import NeuralNetRegressor
import matplotlib.pyplot as plt
import gc
import re
from sklearn.inspection import permutation_importance

import sys
sys.path.insert(1, '/project/cper_neon_aop/hls_nrt/extract')
sys.path.insert(1, '/project/cper_neon_aop/hls_nrt/fit')
from dnn_setup import ResNetRegressor, ResidualBlock

import warnings
from sklearn.exceptions import ConvergenceWarning

lr_mod = pd.compat.pickle_compat.load(open("/project/cper_neon_aop/cper_hls_veg_models/models/biomass/CPER_HLS_to_VOR_biomass_model_lr_simp.pk", 'rb'))

cper_var_dict = {
    'NDVI': 'ndvi',
    'DFI': 'dfi',
    'NDTI': 'ndti',
    'SATVI': 'satvi',
    'NDII7': 'ndii7',
    'SAVI': 'savi',
    'RDVI': 'rdvi',
    'MTVI1': 'mtvi1', 
    'NCI': 'nci', 
    'NDCI': 'ndci',
    'PSRI': 'psri',
    'NDWI': 'ndwi',
    'EVI': 'evi',
    'TCBI': 'tcbi',
    'TCGI': 'tcgi',
    'TCWI': 'tcwi',
    'BAI_126': 'bai_126',
    'BAI_136': 'bai_136',
    'BAI_146': 'bai_146',
    'BAI_236': 'bai_236',
    'BAI_246': 'bai_246',
    'BAI_346': 'bai_346',
    'BLUE': 'blue',
    'GREEN': 'green',
    'RED': 'red',
    'NIR1': 'nir',
    'SWIR1': 'swir1',
    'SWIR2': 'swir2'
}

def logy(x):
    return np.log(1 + x)

def logy_bt(x):
    return np.exp(x) - 1

def sqrty(x):
    return np.sqrt(x)

def sqrty_bt(x):
    return x**2

def make_model_dictionary(var_names, y_col, device):
    
    rnr = NeuralNetRegressor(
        ResNetRegressor(ResidualBlock, layers=[3, 4, 6, 3], n_inputs=len(var_names)),
        criterion=nn.L1Loss,
        optimizer=optim.SGD,
        lr=0.001,
        max_epochs=100,
        batch_size=64,
        train_split=skorch.dataset.ValidSplit(0.2),
        callbacks=[skorch.callbacks.EarlyStopping(patience=10, load_best=True)],
        verbose=0, 
        device=device
    )
    
    mod_dict = {
        'CPER_2022': {
            'base_mod': lr_mod,
            'fit': True,
            'tune': False,
            'tune_refit': None,
            'tune_refit_type': None,
            'variable_importance': False,
            'scale_x': False,
            'xfrm_y': logy,
            'bxfrm_y': logy_bt,
            'interactions': False,
            'interaction_only': True,
            'interaction_poly': 2,
        },
        'OLS_2022': {
            'base_mod': y_col + ' ~ NDII7 + NIR1 + BAI_236 + NDII7:NIR1 + NDII7:BAI_236 + NIR1:BAI_236',
            'fit': True,
            'tune': True,
            'tune_refit': 'mae_orig_mean',
            'tune_refit_type': 'minimize',
            'variable_importance': False,
            'scale_x': False,
            'xfrm_y': sqrty,
            'bxfrm_y': sqrty_bt,
            'interactions': False,
            'interaction_only': True,
            'interaction_poly': 2,
        },
        'OLS': {
            'base_mod': y_col + ' ~ ',
            'fit': True,
            'tune': True,
            'tune_refit': 'mae_orig_mean',
            'tune_refit_type': 'minimize',
            'variable_importance': False,
            'scale_x': False,
            'xfrm_y': sqrty,
            'bxfrm_y': sqrty_bt,
            'interactions': False,
            'interaction_only': True,
            'interaction_poly': 2,
            'formula_df': pd.DataFrame(columns=['kfold', 'kfold_name', 'numb_vars', 'formula', 'R2_adj', 'AIC', 'mae_orig_mean'])
        },
        'LASSO': {
            'base_mod': Pipeline(
                [
                    ('scaler', StandardScaler()), 
                    ('LASSO', Lasso())
                ]),
            'fit': True,
            'tune': True,
            'variable_importance': True,
            'tune_refit': 'MAE',
            'param_grid': {
                'LASSO__alpha': np.logspace(-3, 1, num=30)
            },
            'tune_results': {},
            'scale_x': False,
            'scaler': StandardScaler(),
            'xfrm_y': sqrty,
            'bxfrm_y': sqrty_bt,
            'interactions': True,
            'interaction_only': True,
            'interaction_poly': 2,
            'variable_df': pd.DataFrame(columns=['kfold', 'Variable', 'Coef'])
        },
        'PLS': {
            'base_mod': Pipeline(
                [
                    ('scaler', StandardScaler()), 
                    ('PLS', PLSRegression(n_components=1, scale=False))
                ]),
            'fit': True,
            'tune': True,
            'tune_vip': False,
            'tune_vip_iters': 3,
            'tune_vip_thresh': [0.8, 0.8, 0.8],
            'variable_importance': True,
            'tune_refit': 'MAE',
            'param_grid': {
                'PLS__n_components': [int(x) for x in np.arange(1, len(var_names)*0.5)]
            },
            'tune_results': {},
            'scale_x': False,
            'scaler': StandardScaler(),
            'xfrm_y': sqrty,
            'bxfrm_y': sqrty_bt,
            'interactions': False,
            'interaction_only': True,
            'interaction_poly': 2,
            'variable_df': pd.DataFrame(columns=['kfold', 'Variable', 'VIP', 'Coef'])
        },
        'PCR': {
            'base_mod': Pipeline(
                [
                    ('scaler', StandardScaler()), 
                    ('pca', PCA()), 
                    ('linreg', LinearRegression())
                ]),
            'fit': True,
            'tune': True,
            'variable_importance': True,
            'tune_refit': 'MAE',
            'param_grid': {
                'pca__n_components': [int(x) for x in np.arange(1, len(var_names)*0.5)]
            },
            'tune_results': {},
            'scale_x': False,
            'scaler': StandardScaler(),
            'xfrm_y': sqrty,
            'bxfrm_y': sqrty_bt,
            'interactions': False,
            'interaction_only': True,
            'interaction_poly': 2,
            'variable_df': pd.DataFrame(columns=['kfold', 'Variable'] + ['PC_' + str(i+1) for i in range(len(var_names))])
        },
        'SVR': {
            'base_mod': Pipeline(
                [
                    ('scaler', StandardScaler()), 
                    ('SVR', LinearSVR(dual='auto'))
                    #('SVR', SVR(kernel='linear', cache_size=1000))
                ]),
            'fit': True,
            'variable_importance': True,
            'tune': True,
            'tune_refit': 'MAE',
            'param_grid': {
                'SVR__C': np.logspace(0.1, 3.0, 5, base=10),
                'SVR__epsilon': [0.0, 0.01, 0.1],
                #'SVR__C': np.logspace(0.1, 3.0, 10, base=10),
                #'SVR__gamma': np.logspace(-3.5, 0, 10, base=10)
            },
            'tune_results': {},
            'scale_x': False,
            'scaler': StandardScaler(),
            'xfrm_y': sqrty,
            'bxfrm_y': sqrty_bt,
            'interactions': False,
            'interaction_only': True,
            'interaction_poly': 2,
            'variable_df': pd.DataFrame(columns=['kfold', 'Variable', 'SVR_weights'])
        },
        'RF': {
            'base_mod': Pipeline(
                [
                    ('scaler', StandardScaler()), 
                    ('RF', RandomForestRegressor(bootstrap=True, oob_score=True, n_jobs=-1))
                ]),
            'fit': True,
            'tune': True,
            'variable_importance': True,
            'tune_refit': 'MAE',
            'param_grid': {
                'RF__min_samples_split': [0.001, 0.01, 0.10],
                #'RF__n_estimators': [200, 400],
                'RF__n_estimators': [50, 100],
                'RF__max_samples': [0.25, 0.5, 0.75, 0.90],
                'RF__max_features': [0.1, 0.25, 0.5, 0.75]
                #'RF__max_features': [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
            },
            'tune_results': {},
            'scale_x': False,
            'scaler': StandardScaler(),
            'xfrm_y': sqrty,
            'bxfrm_y': sqrty_bt,
            'interactions': False,
            'interaction_only': True,
            'interaction_poly': 2,
            'variable_df': pd.DataFrame(columns=['kfold', 'Variable', 'MDI'])
        },
        'GBR': {
            #'base_mod': XGBRegressor(n_jobs=-1, verbosity=0),
            'base_mod': Pipeline(
                [
                    ('scaler', StandardScaler()), 
                    ('GBR', GradientBoostingRegressor(loss='absolute_error'))
                ]),
            'fit': False,
            'variable_importance': True,
            'tune': True,
            'tune_refit': 'MAE',
            'param_grid': {
                'GBR__learning_rate': [0.1, 0.05, 0.025, 0.01, 0.001],
                'GBR__min_samples_split': [0.001, 0.01, 0.10],
                'GBR__n_estimators': [200, 400, 600, 800],
                #'GBR__max_features': [0.1, 0.25, 0.5]
            },
            'tune_results': {},
            'scale_x': False,
            'scaler': StandardScaler(),
            'xfrm_y': sqrty,
            'bxfrm_y': sqrty_bt,
            'interactions': False,
            'interaction_only': True,
            'interaction_poly': 2,
            'variable_df': pd.DataFrame(columns=['kfold', 'Variable', 'MDI'])
        },
        'HGBR': {
            #'base_mod': XGBRegressor(n_jobs=-1, verbosity=0),
            'base_mod': Pipeline(
                [
                    ('scaler', StandardScaler()), 
                    ('GBR', HistGradientBoostingRegressor())
                ]),
            'fit': True,
            'variable_importance': False,
            'tune': True,
            'tune_refit': 'MAE',
            'param_grid': {
                'GBR__learning_rate': [0.1, 0.05, 0.025, 0.01, 0.001],
                'GBR__min_samples_leaf': [20, 50, 100],
                #'GBR__n_estimators': [200, 400, 600, 800],
                'GBR__max_features': [0.1, 0.25, 0.5]
            },
            'tune_results': {},
            'scale_x': False,
            'scaler': StandardScaler(),
            'xfrm_y': sqrty,
            'bxfrm_y': sqrty_bt,
            'interactions': False,
            'interaction_only': True,
            'interaction_poly': 2,
            'variable_df': pd.DataFrame(columns=['kfold', 'Variable', 'MDI'])
        },
        'MLP': {
            'base_mod': MLPRegressor(solver='adam', activation='logistic', hidden_layer_sizes=(256,),
                                     max_iter=1000, learning_rate='adaptive'),
            'fit': False,
            'variable_importance': False,
            'tune': False,
            'tune_refit': 'MAE',
            'param_grid': {
                'hidden_layer_sizes': [(50,), (100,), (150,)],
                'alpha': [0.00005, 0.0001, 0.0005, 0.001]
            },
            'tune_results': {},
            'scale_x': True,
            'scaler': StandardScaler(),
            'xfrm_y': sqrty,
            'bxfrm_y': sqrty_bt,
            'interactions': False,
            'interaction_only': True,
            'interaction_poly': 2,
            'variable_df': pd.DataFrame()
        },
        'DNN': {
            'base_mod': rnr,
            'fit': False,
            'variable_importance': False,
            'tune': True,
            'tune_refit': 'MAE',
            'param_grid': {
                'module__block': [ResidualBlock],
                'module__layers': [[3, 4, 6, 3]], 
                'module__n_inputs': [len(var_names)],
                'optimizer__momentum': [0.95],
                'optimizer__weight_decay': [1e-2],
                'optimizer__nesterov': [True],
                'lr': [0.1, 0.05, 0.025, 0.01, 0.001],
                'batch_size': [64],
                'max_epochs': [100],
            },
            'tune_results': {},
            'scale_x': True,
            'scaler': MinMaxScaler(),
            'xfrm_y': sqrty,
            'bxfrm_y': sqrty_bt,
            'interactions': False,
            'interaction_only': True,
            'interaction_poly': 2,
            'variable_df': pd.DataFrame()
    }
    }

    return mod_dict


def vip(x, y, model):
    t = model.x_scores_
    #w = model.x_weights_
    w = model.x_rotations_
    q = model.y_loadings_

    m, p = x.shape
    _, h = t.shape

    vips = np.zeros((p,))

    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)

    for i in range(p):
        weight = np.array([ (w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h) ])
        vips[i] = np.sqrt(p*(s.T @ weight)/total_s)

    return vips

@dask.delayed
def fit_ols(all_x, mod_split, split_groups, df_train, y_col, lr_form, group_k, kfold_group, k_fold, idx):
    r2_adj_tmp = []
    aic_tmp = []
    mae_orig_tmp = []
    for train_index_sub, test_index_sub in mod_split.split(all_x, groups=split_groups):
        df_train_sub = df_train.iloc[train_index_sub]
        df_test_sub = df_train.iloc[test_index_sub]
        lreg_k_tmp = smf.ols(formula=lr_form, data=df_train_sub).fit()
        r2_adj_tmp.append(lreg_k_tmp.rsquared_adj)
        aic_tmp.append(lreg_k_tmp.aic)
        mae_orig_tmp.append(np.nanmean(np.abs(lreg_k_tmp.predict(df_test_sub) - df_test_sub[y_col])))
    df_results_tmp = pd.DataFrame(dict(kfold=[group_k],
                                           kfold_name=[kfold_group],
                                           numb_vars=[k_fold],
                                           formula=[lr_form],
                                           R2_adj=round(np.mean(r2_adj_tmp), 4),
                                           AIC=round(np.mean(aic_tmp), 4),
                                           mae_orig_mean=round(np.mean(mae_orig_tmp), 4)),
                                     index=[idx])
    return df_results_tmp

@dask.delayed
def fit_dnn(mod_base, batch_start, batch_size, all_x, all_y, loss_fn, optimizer):
    for start in batch_start:
        # take a batch
        X_batch = all_x[start:start+mod_dict[k]['batch_size']]
        y_batch = all_y[start:start+mod_dict[k]['batch_size']]
        # forward pass
        y_pred = mod_base(X_batch)
        loss = loss_fn(y_pred, y_batch)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()
    return mod_base


def run_ml_models(nickname, mod_dict, df, y_col, date_col, var_names, kfold_group, tuneby_group, kfold_type, tune_kfold_type, outFILE_tmp, outDIR,
                  backend, nthreads,
                  cper_mod_xfrm, cper_mod_xfrm_func, client,
                  cper_var_dict=cper_var_dict,
                  n_splits=10):
    if os.path.exists(outFILE_tmp):
        print('Output file already exists. Loading saved dataset.')
        df = pd.read_csv(outFILE_tmp, parse_dates=[date_col])
        with open(os.path.join(outDIR, 'tmp', 'ml_train_' + nickname + '_cv_' + kfold_group + '_tuneby_' + tuneby_group + '_results.pk'), 'rb') as f:
            mod_dict = pd.compat.pickle_compat.load(f)
    else:
        for k in mod_dict:
            if mod_dict[k]['fit']:
                df[k] = np.nan
        if kfold_type == 'group_k':
            df['kfold'] = ''
    
    mod_logo = LeaveOneGroupOut()
    mod_groupk = GroupKFold(n_splits=n_splits)
    
    if kfold_type == 'logo':
        mod_split = mod_logo
    elif kfold_type == 'group_k':
        mod_split = mod_groupk
        kfold = 0
        
    scoring = {
        #'R2': 'r2',
        #'MSE': 'neg_mean_squared_error',
        'MAE': 'neg_mean_absolute_error',
        #'MAPE': 'neg_mean_absolute_percentage_error',
        #'MSLE': 'neg_mean_squared_log_error'
    }

    if backend == 'dask':
        from dask_ml.model_selection import GridSearchCV
        from sklearn.model_selection import GridSearchCV as skGridSearchCV
    else:
        from sklearn.model_selection import GridSearchCV
    
    restart_dask = True
    
    for train_index, test_index in mod_split.split(df[var_names], groups=df[kfold_group], ):
        logo_test = df[kfold_group].iloc[test_index].unique()
        if kfold_type == 'logo':
            logo_k = logo_test[0]
        elif kfold_type == 'group_k':
            kfold += 1
            logo_k = 'kfold' + str(kfold)
            
        print(logo_k)
        train_loc = df.iloc[train_index].index
        test_loc = df.iloc[test_index].index
        
        all_y_orig = df[y_col].iloc[train_index]
        all_Y_orig = df[y_col].iloc[test_index]
        all_x_orig = df[var_names].iloc[train_index, :]
        all_X_orig = df[var_names].iloc[test_index, :]
    
        for k in mod_dict:
            if mod_dict[k]['fit']:
                if df[df[kfold_group].isin(logo_test)][k].isnull().all():
                    restart_dask = False
                    print('....fitting ' + k, end = " ")
                    t0 = time.time()
                    
                    # prep data
                    if mod_dict[k]['xfrm_y'] is not None:
                        all_y = all_y_orig.apply(mod_dict[k]['xfrm_y'])     
                        all_Y = all_Y_orig.apply(mod_dict[k]['xfrm_y'])
                    else:
                        all_y = all_y_orig.copy()
                        all_Y = all_Y_orig.copy()
                    if mod_dict[k]['scale_x']:
                        scaler = mod_dict[k]['scaler']
                        scaler.fit(all_x_orig)
                        all_x = scaler.transform(all_x_orig)
                        all_X = scaler.transform(all_X_orig)
                    else:
                        all_x = all_x_orig.copy()
                        all_X = all_X_orig.copy()
                    
                    if mod_dict[k]['interactions']:
                        poly_x = PolynomialFeatures(degree=mod_dict[k]['interaction_poly'], 
                                                    interaction_only=mod_dict[k]['interaction_only'], include_bias = False)
                        all_x = poly_x.fit_transform(all_x)
                        poly_X = PolynomialFeatures(degree=mod_dict[k]['interaction_poly'], 
                                                    interaction_only=mod_dict[k]['interaction_only'], include_bias = False)
                        all_X = poly_X.fit_transform(all_X)
                        var_names_out = poly_x.get_feature_names_out(var_names)
                    else:
                        var_names_out = var_names
        
                    if k == 'DNN':
                        all_x = all_x.astype('float32')
                        all_y = all_y.astype('float32').values.reshape(-1, 1)
                        all_X = all_X.astype('float32')
                        all_Y = all_Y.astype('float32')
                        #client.scatter([DNNRegressor, dnnr], broadcast=True)
                    
                    # create a base model
                    mod_base = mod_dict[k]['base_mod']
                    
                    if mod_dict[k]['tune']:
                        split_groups = df[tuneby_group].iloc[train_index]
                        if tune_kfold_type == 'logo':
                            cv_splitter = mod_logo.split(all_x, groups=split_groups)
                        elif tune_kfold_type == 'group_k':
                            cv_splitter = mod_groupk.split(all_x, groups=split_groups)
    
                        if 'OLS' in k:
                            df_train = pd.merge(pd.DataFrame(data=all_y),
                                                pd.DataFrame(columns=all_x_orig.columns, data=all_x, index=all_x_orig.index),
                                                left_index=True,
                                                right_index=True)
                            df_test = pd.merge(pd.DataFrame(data=all_Y),
                                                pd.DataFrame(columns=all_X_orig.columns, data=all_X, index=all_X_orig.index),
                                                left_index=True,
                                                right_index=True)
                            if k == 'OLS_2022':
                                form_fnl = mod_dict[k]['base_mod']
                            else:
                                idx = 0
                                df_results_list = []
                                for k_fold in range(1, 3 + 1):
                                    for combo in itertools.combinations(var_names, k_fold):
                                        combo_corr = df[np.array(combo)].corr()
                                        if ((combo_corr != 1.0) & (combo_corr.abs() > 0.8)).any(axis=None):
                                            continue
                                        else:
                                            lr_form = mod_dict[k]['base_mod'] + combo[0]
                                            if k_fold > 1:
                                                for c in combo[1:]:
                                                    lr_form = lr_form + ' + ' + c
                                                for combo_c in itertools.combinations(combo, 2):
                                                    lr_form = lr_form + ' + ' + combo_c[0] + ':' + combo_c[1]
                                            df_results_tmp = fit_ols(all_x,
                                                                     mod_split, 
                                                                     split_groups,
                                                                     df_train,
                                                                     y_col,
                                                                     lr_form, 
                                                                     logo_k,
                                                                     kfold_group, 
                                                                     k_fold,
                                                                     idx)
                                            df_results_list.append(df_results_tmp)
                                            #mod_dict[k]['formula_df'] = pd.concat([df_results_tmp.compute(), mod_dict[k]['formula_df']])
                                            #break
                                df_results = dask.compute(df_results_list)
                                mod_dict[k]['formula_df'] = pd.concat([mod_dict[k]['formula_df'], pd.concat(df_results[0])])
                                if mod_dict[k]['tune_refit_type'] == 'minimize':
                                    tune_loc = 0
                                elif mod_dict[k]['tune_refit_type'] == 'maximize':
                                    tune_loc = -1
                                form_fnl = mod_dict[k]['formula_df'][mod_dict[k]['formula_df']['kfold'] == logo_k].sort_values(
                                    mod_dict[k]['tune_refit'])['formula'].iloc[tune_loc]
                            mod_fnl = smf.ols(formula=form_fnl, data=df_train).fit()
                        
                        elif k == 'MLP':
                            from sklearn.model_selection import GridSearchCV as skGridSearchCV
                            grid_search = skGridSearchCV(estimator=mod_base,
                                                           param_grid=mod_dict[k]['param_grid'],
                                                           scoring=scoring, 
                                                           refit=mod_dict[k]['tune_refit'], 
                                                           return_train_score=True,
                                                           cv=cv_splitter, 
                                                           n_jobs=min(sum([len(x) for x in mod_dict[k]['param_grid']]),
                                                                      len(client.nthreads())))
                            with parallel_backend('threading'):
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore", category=ConvergenceWarning)
                                    grid_search.fit(all_x, all_y)
                            mod_fnl = mod_base.set_params(**grid_search.best_params_)
                            mod_fnl.fit(all_x, all_y)
                            mod_dict[k]['tune_results'][logo_k] = grid_search.cv_results_
                        elif k == 'DNN':
                            from sklearn.model_selection import GridSearchCV as skGridSearchCV
                            grid_search = skGridSearchCV(estimator=mod_base,
                                                         param_grid=mod_dict[k]['param_grid'],
                                                         scoring=scoring, 
                                                         refit=mod_dict[k]['tune_refit'], 
                                                         return_train_score=True,
                                                         cv=cv_splitter, 
                                                         n_jobs=min(sum([len(x) for x in mod_dict[k]['param_grid']]),
                                                                      len(client.nthreads())))
                            grid_search.fit(all_x, all_y)
                            mod_fnl = mod_base.set_params(**grid_search.best_params_)
                            mod_fnl.fit(all_x, all_y)
                            ax = plt.subplot()
                            p_vl, = ax.plot(mod_fnl.history[:, 'valid_loss'], label='Validation')
                            p_tl, = ax.plot(mod_fnl.history[:, 'train_loss'], label='Training')
                            ax.legend(handles=[p_vl, p_tl])
                            plt.show()
                            mod_dict[k]['tune_results'][logo_k] = grid_search.cv_results_
                        else:
                            grid_search = GridSearchCV(estimator=mod_base,
                                                           param_grid=mod_dict[k]['param_grid'],
                                                           scoring=scoring, 
                                                           refit=mod_dict[k]['tune_refit'], 
                                                           return_train_score=True,
                                                           cv=cv_splitter, 
                                                           n_jobs=min(sum([len(x) for x in mod_dict[k]['param_grid']]),
                                                                     nthreads))
                            with parallel_backend(backend):
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore", category=ConvergenceWarning)
                                    grid_search.fit(all_x, all_y)
                            mod_fnl = mod_base.set_params(**grid_search.best_params_)
                            mod_fnl.fit(all_x, all_y)
                            mod_dict[k]['tune_results'][logo_k] = grid_search.cv_results_
                    else:
                        if k == 'CPER_2022':
                            mod_fnl = lr_mod
                            all_x = all_x.rename(columns=cper_var_dict)
                            all_X = all_X.rename(columns=cper_var_dict)
                        elif k == 'DNN':
                            mod_fnl = mod_base
                            mod_fnl.fit(all_x, all_y)
                            ax = plt.subplot()
                            p_vl, = ax.plot(mod_fnl.history[:, 'valid_loss'], label='Validation')
                            p_tl, = ax.plot(mod_fnl.history[:, 'train_loss'], label='Training')
                            ax.legend(handles=[p_vl, p_tl])
                            plt.show()
                            cp = skorch.callbacks.Checkpoint(dirname='results/dnn_checkpoints')
                            mod_fnl.initialize()
                            mod_fnl.load_params(checkpoint=cp)
                        else:
                            mod_fnl = mod_base
                            mod_fnl.fit(all_x, all_y)
        
                    if mod_dict[k]['variable_importance']:
                        if k == 'LASSO':
                            lasso_coefs = abs(mod_fnl[k].coef_)
                            var_names_out = [x for idx, x in enumerate(var_names_out) if abs(lasso_coefs[idx]) != 0.0]
                            lasso_coefs = lasso_coefs[abs(lasso_coefs) > 0.0]
                            mod_dict[k]['variable_df'] = pd.concat([mod_dict[k]['variable_df'],
                                                                    pd.DataFrame({'kfold': logo_k,
                                                                                  'Variable': var_names_out,
                                                                                  'Coef': lasso_coefs})])
                        if k == 'PLS':
                            pls_vip = vip(all_x, all_y, mod_fnl[k])
                            pls_coefs = abs(mod_fnl[k].coef_).squeeze()
                            if mod_dict[k]['tune_vip']:
                                if len(mod_dict[k]['tune_vip_thresh']) != mod_dict[k]['tune_vip_iters']:
                                    print('ERROR: Length of tune_vip_thresh does not equal tune_vip_iters.')
                                    break
                                else:
                                    for i in range(mod_dict[k]['tune_vip_iters']):
                                        vip_thresh = mod_dict[k]['tune_vip_thresh'][i]
                                        all_x = all_x[:, pls_vip > vip_thresh]
                                        all_X = all_X[:, pls_vip > vip_thresh]
                                        mod_fnl.fit(all_x, all_y)
                                        var_names_out = [x for idx, x in enumerate(var_names_out) if pls_vip[idx] > vip_thresh]
                                        pls_vip = vip(all_x, all_y, mod_fnl)
                                        pls_coefs = abs(mod_fnl.coef_).squeeze()
                                
                            mod_dict[k]['variable_df'] = pd.concat([mod_dict[k]['variable_df'],
                                                                    pd.DataFrame({'kfold': logo_k,
                                                                                  'Variable': var_names_out,
                                                                                  'VIP': pls_vip,
                                                                                  'Coef': pls_coefs})])
                        if k == 'PCR':
                            # get distributed coefficients by multiplying variable loadings by PC coefficients
                            coefs = pd.DataFrame(mod_fnl['pca'].components_.T  * mod_fnl['linreg'].coef_, 
                                                    columns=['PC_' + str(i+1) for i in range(grid_search.best_params_['pca__n_components'])], 
                                                    index=var_names_out).reset_index().rename(columns={'index': 'Variable'})
                            coefs['kfold'] = logo_k
                            mod_dict[k]['variable_df'] = pd.concat([mod_dict[k]['variable_df'],
                                                                    coefs])
                            
                        if k == 'SVR':
                            svm_weights = mod_fnl[k].coef_
                            mod_dict[k]['variable_df'] = pd.concat([mod_dict[k]['variable_df'],
                                                                    pd.DataFrame({'kfold': logo_k,
                                                                                  'Variable': var_names_out,
                                                                                  'SVR_weights': svm_weights.squeeze()})])
                        if k in ['RF', 'GBR']:
                            mdi = mod_fnl[k].feature_importances_
                            mod_dict[k]['variable_df'] = pd.concat([mod_dict[k]['variable_df'],
                                                                    pd.DataFrame({'kfold': logo_k,
                                                                                  'Variable': var_names_out,
                                                                                  'MDI': mdi})])
                        if k in ['HGBR']:
                            rf_pi = permutation_importance(mod_fnl, all_X, all_Y, n_repeats=10, n_jobs=-1)
                            mod_dict[k]['variable_df'] = pd.concat([mod_dict[k]['variable_df'],
                                                                    pd.DataFrame({'kfold': logo_k,
                                                                                  'Variable': var_names_out,
                                                                                  'PI': rf_pi.importances_mean})])
                    
                    if mod_dict[k]['bxfrm_y'] is not None:
                        if mod_dict[k] == 'OLS':
                            preds = mod_fnl.predict(df_test)
                            preds[preds < 0] = 0
                            preds = mod_dict[k]['bxfrm_y'](preds)
                        else:
                            preds = mod_fnl.predict(all_X).squeeze()
                            preds[preds < 0] = 0
                            preds = mod_dict[k]['bxfrm_y'](preds)
                    else:
                        if mod_dict[k] == 'OLS':
                            preds = mod_fnl.predict(df_test)
                            preds[preds < 0] = 0
                        else:
                            preds = mod_fnl.predict(all_X).squeeze()
                            preds[preds < 0] = 0
    
                    # apply transformation to CPER 2022 model
                    if (k == 'CPER_2022') and (cper_mod_xfrm):
                        preds = cper_mod_xfrm_func(preds)
                        
                    df.loc[test_loc, k] = preds
                    df.loc[test_loc, 'kfold'] = logo_k
                    
                    print('(time to fit: ' + str(round(time.time() - t0, 2)) + ' secs)')
        
                    # save temporary file to disk
                    df.to_csv(outFILE_tmp, index=False)
        
                    with open(os.path.join(outDIR, 'tmp', 'ml_train_' + nickname + '_cv_' + kfold_group + '_tuneby_' + tuneby_group + '_results.pk'), 'wb') as fp:
                        pickle.dump(mod_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
                    if mod_dict[k]['tune'] and (mod_dict[k] in ['SVR', 'RF', 'GBR']):
                        del grid_search, mod_fnl
                        gc.collect()
                else:
                    restart_dask = False
                    print('Skipping ' + k + ', data already in saved dataframe.')
                    continue
            else:
                print('Skipping ' + k + ', params not set to fit.')
                continue
        if backend == 'dask' and mod_dict['DNN']['fit'] and restart_dask:
            # restart client
            client.restart(wait_for_workers=False)
            # make sure there are at least some workers before fetching data
            client.wait_for_workers(n_workers=num_jobs*num_processes*0.35, timeout=300)

def r_corrcoef(y_obs, y_pred):
    corr_matrix = np.corrcoef(y_obs, y_pred)
    corr = corr_matrix[0,1]
    r = corr
    return r

def run_ml_models_bootstrap_yr(nickname, mod_dict, df, y_col, date_col, var_names,
                               tuneby_group,
                               backend, nthreads,
                               cper_mod_xfrm, cper_mod_xfrm_func, client,
                               cper_var_dict=cper_var_dict,
                               retune_bootstrap=True,
                               agg_plot=False,
                               save_path=None):
    from sklearn.metrics import r2_score
    import itertools
    from tqdm import tqdm

    if save_path is None:
        df_results_yrs = None
    else:
        if os.path.exists(save_path):
            df_results_yrs = pd.read_csv(save_path)
        else:
            df_results_yrs = None
    
    mod_logo = LeaveOneGroupOut()
    mod_split = mod_logo

    scoring = {
        'MAE': 'neg_mean_absolute_error',
    }
    if backend == 'dask':
        from dask_ml.model_selection import GridSearchCV
        from sklearn.model_selection import GridSearchCV as skGridSearchCV
    else:
        from sklearn.model_selection import GridSearchCV
    
    idx_ct = 0
    for yr_n in range(3, 1 + len(df['Year'].unique())):
        print('Running ' + str(yr_n) + '-year combos')
        combos = list(itertools.combinations(df['Year'].unique(), yr_n))
        print(len(combos))
        if df_results_yrs is not None:
            nyrs_complete = df_results_yrs['numb_yrs'].unique()
            check_nyrs_lens = all(
                df_results_yrs[df_results_yrs['numb_yrs'] == yr_n - 1].groupby('Model').count()['numb_yrs']/yr_n == len(combos))
        else:
            nyrs_complete = []
            check_nyrs_lens = False
        if ((yr_n - 1) in nyrs_complete) and check_nyrs_lens:
            print('All iterations run for all models. Skipping combos.')
            continue
        else:
            for yr_combo in tqdm(combos):
                df_sub = df[df['Year'].isin(yr_combo)]
                df_sub = df_sub.dropna(subset=var_names + [y_col])
                t0 = time.time()
                
                for train_index, test_index in mod_logo.split(df_sub, groups=df_sub['Year']):
                    yr = df_sub[date_col].dt.year.iloc[test_index].unique()[0]
                    #print(str(df_sub[date_col].dt.year.iloc[train_index].unique()))
                    #print(len(str(df_sub[date_col].dt.year.iloc[train_index].unique())))
                    #df_results_yrs = df_results_yrs.reset_index(drop=True)
                    train_yrs_tmp = re.sub(',',
                                           '', 
                                           str(df_sub[date_col].dt.year.iloc[train_index].unique()))
                    if df_results_yrs is not None:
                        outlen_tmp = sum(df_results_yrs['yr_train'].isin([train_yrs_tmp]) & (df_results_yrs['yr_test'] == yr))
                        outlen_check = outlen_tmp == sum([int(mod_dict[i]['fit']) for i in mod_dict.keys()])
                    else:
                        outlen_check = False
                    if outlen_check:
                        continue
                    else:
                        logo_k = yr
                        train_loc = df_sub.iloc[train_index].index
                        test_loc = df_sub.iloc[test_index].index
                        
                        all_y_orig = df_sub[y_col].iloc[train_index]
                        all_Y_orig = df_sub[y_col].iloc[test_index]
                        all_x_orig = df_sub[var_names].iloc[train_index, :]
                        all_X_orig = df_sub[var_names].iloc[test_index, :]
                    
                        for k in mod_dict:
                            if mod_dict[k]['fit']:
                                #print('....fitting ' + k, end = " ")
                                run_check = k + ' '.join([str(x) for x in df_sub[date_col].dt.year.iloc[train_index].unique()])+str(yr)
                                if df_results_yrs is not None:
                                    run_check_result = run_check in df_results_yrs.apply(
                                        lambda x: x['Model']+''.join([str(i) for i in x['yr_train'] if i not in ['[', ']']])+str(x['yr_test']),
                                        axis=1).values
                                if df_results_yrs is not None and run_check_result:
                                        #print('Skipping ' + run_check + '. Already saved.')
                                        continue
                                else:
                                    # prep data
                                    if mod_dict[k]['xfrm_y'] is not None:
                                        all_y = all_y_orig.apply(mod_dict[k]['xfrm_y'])     
                                        all_Y = all_Y_orig.apply(mod_dict[k]['xfrm_y'])
                                    else:
                                        all_y = all_y_orig.copy()
                                        all_Y = all_Y_orig.copy()
                                    if mod_dict[k]['scale_x']:
                                        scaler = mod_dict[k]['scaler']
                                        scaler.fit(all_x_orig)
                                        all_x = scaler.transform(all_x_orig)
                                        all_X = scaler.transform(all_X_orig)
                                    else:
                                        all_x = all_x_orig.copy()
                                        all_X = all_X_orig.copy()
                                    
                                    if mod_dict[k]['interactions']:
                                        poly_x = PolynomialFeatures(degree=mod_dict[k]['interaction_poly'], 
                                                                    interaction_only=mod_dict[k]['interaction_only'], include_bias = False)
                                        all_x = poly_x.fit_transform(all_x)
                                        poly_X = PolynomialFeatures(degree=mod_dict[k]['interaction_poly'], 
                                                                    interaction_only=mod_dict[k]['interaction_only'], include_bias = False)
                                        all_X = poly_X.fit_transform(all_X)
                                        var_names_out = poly_x.get_feature_names_out(var_names)
                                    else:
                                        var_names_out = var_names
                        
                                    # create a base model
                                    mod_base = mod_dict[k]['base_mod']
                                    # set parameters
                                    if retune_bootstrap:
                                        split_groups = df[tuneby_group].iloc[train_index]
                                        cv_splitter = mod_logo.split(all_x, groups=split_groups)
            
                                        if 'OLS' in k:
                                            df_train = pd.merge(pd.DataFrame(data=all_y),
                                                                pd.DataFrame(columns=all_x_orig.columns, data=all_x, index=all_x_orig.index),
                                                                left_index=True,
                                                                right_index=True)
                                            df_test = pd.merge(pd.DataFrame(data=all_Y),
                                                                pd.DataFrame(columns=all_X_orig.columns, data=all_X, index=all_X_orig.index),
                                                                left_index=True,
                                                                right_index=True)
                                            if k == 'OLS_2022':
                                                form_fnl = mod_dict[k]['base_mod']
                                            else:
                                                idx = 0
                                                df_results_list = []
                                                for k_fold in range(1, 3 + 1):
                                                    for combo in itertools.combinations(var_names, k_fold):
                                                        combo_corr = df[np.array(combo)].corr()
                                                        if ((combo_corr != 1.0) & (combo_corr.abs() > 0.8)).any(axis=None):
                                                            continue
                                                        else:
                                                            lr_form = mod_dict[k]['base_mod'] + combo[0]
                                                            if k_fold > 1:
                                                                for c in combo[1:]:
                                                                    lr_form = lr_form + ' + ' + c
                                                                for combo_c in itertools.combinations(combo, 2):
                                                                    lr_form = lr_form + ' + ' + combo_c[0] + ':' + combo_c[1]
                                                            df_results_tmp = fit_ols(all_x,
                                                                                     mod_split, 
                                                                                     split_groups,
                                                                                     df_train,
                                                                                     y_col,
                                                                                     lr_form, 
                                                                                     yr,
                                                                                     logo_k, 
                                                                                     k_fold,
                                                                                     idx)
                                                            df_results_list.append(df_results_tmp)
                                                            #mod_dict[k]['formula_df'] = pd.concat([df_results_tmp.compute(), mod_dict[k]['formula_df']])
                                                            #break
                                                df_results = dask.compute(df_results_list)
                                                mod_dict[k]['formula_df'] = pd.concat([mod_dict[k]['formula_df'], pd.concat(df_results[0])])
                                                if mod_dict[k]['tune_refit_type'] == 'minimize':
                                                    tune_loc = 0
                                                elif mod_dict[k]['tune_refit_type'] == 'maximize':
                                                    tune_loc = -1
                                                form_fnl = mod_dict[k]['formula_df'][mod_dict[k]['formula_df']['kfold'] == logo_k].sort_values(
                                                    mod_dict[k]['tune_refit'])['formula'].iloc[tune_loc]
                                            mod_fnl = smf.ols(formula=form_fnl, data=df_train).fit()
                                        
                                        elif k == 'MLP':
                                            from sklearn.model_selection import GridSearchCV as skGridSearchCV
                                            grid_search = skGridSearchCV(estimator=mod_base,
                                                                           param_grid=mod_dict[k]['param_grid'],
                                                                           scoring=scoring, 
                                                                           refit=mod_dict[k]['tune_refit'], 
                                                                           return_train_score=True,
                                                                           cv=cv_splitter, 
                                                                           n_jobs=min(sum([len(x) for x in mod_dict[k]['param_grid']]),
                                                                                      len(client.nthreads())))
                                            with parallel_backend('threading'):
                                                with warnings.catch_warnings():
                                                    warnings.simplefilter("ignore", category=ConvergenceWarning)
                                                    grid_search.fit(all_x, all_y)
                                            mod_fnl = mod_base.set_params(**grid_search.best_params_)
                                            mod_fnl.fit(all_x, all_y)
                                            mod_dict[k]['tune_results'][logo_k] = grid_search.cv_results_
                                        elif k == 'DNN':
                                            from sklearn.model_selection import GridSearchCV as skGridSearchCV
                                            grid_search = skGridSearchCV(estimator=mod_base,
                                                                         param_grid=mod_dict[k]['param_grid'],
                                                                         scoring=scoring, 
                                                                         refit=mod_dict[k]['tune_refit'], 
                                                                         return_train_score=True,
                                                                         cv=cv_splitter, 
                                                                         n_jobs=min(sum([len(x) for x in mod_dict[k]['param_grid']]),
                                                                                      len(client.nthreads())))
                                            grid_search.fit(all_x, all_y)
                                            mod_fnl = mod_base.set_params(**grid_search.best_params_)
                                            mod_fnl.fit(all_x, all_y)
                                            ax = plt.subplot()
                                            p_vl, = ax.plot(mod_fnl.history[:, 'valid_loss'], label='Validation')
                                            p_tl, = ax.plot(mod_fnl.history[:, 'train_loss'], label='Training')
                                            ax.legend(handles=[p_vl, p_tl])
                                            plt.show()
                                            mod_dict[k]['tune_results'][logo_k] = grid_search.cv_results_
                                        else:
                                            grid_search = GridSearchCV(estimator=mod_base,
                                                                           param_grid=mod_dict[k]['param_grid'],
                                                                           scoring=scoring, 
                                                                           refit=mod_dict[k]['tune_refit'], 
                                                                           return_train_score=True,
                                                                           cv=cv_splitter, 
                                                                           n_jobs=min(sum([len(x) for x in mod_dict[k]['param_grid']]),
                                                                                     nthreads))
                                            with parallel_backend(backend):
                                                with warnings.catch_warnings():
                                                    warnings.simplefilter("ignore", category=ConvergenceWarning)
                                                    grid_search.fit(all_x, all_y)
                                            mod_fnl = mod_base.set_params(**grid_search.best_params_)
                                            mod_fnl.fit(all_x, all_y)
                                            mod_dict[k]['tune_results'][logo_k] = grid_search.cv_results_
                                    else:
                                        if k == 'OLS_2022':
                                            form_fnl = mod_base
                                            df_train = pd.merge(pd.DataFrame(data=all_y),
                                                            pd.DataFrame(columns=all_x_orig.columns, data=all_x, index=all_x_orig.index),
                                                            left_index=True,
                                                            right_index=True)
                                            mod_fnl = smf.ols(formula=form_fnl, data=df_train).fit()
                                        elif k == 'OLS':
                                            form_fnl = mod_dict[k]['formula_df'].sort_values('mae_orig_mean')['formula'].iloc[0]
                                            df_train = pd.merge(pd.DataFrame(data=all_y),
                                                            pd.DataFrame(columns=all_x_orig.columns, data=all_x, index=all_x_orig.index),
                                                            left_index=True,
                                                            right_index=True)
                                            mod_fnl = smf.ols(formula=form_fnl, data=df_train).fit()
                                        elif k == 'DNN':
                                            mod_fnl = mod_base
                                            mod_fnl.fit(all_x, all_y)
                                            ax = plt.subplot()
                                            p_vl, = ax.plot(mod_fnl.history[:, 'valid_loss'], label='Validation')
                                            p_tl, = ax.plot(mod_fnl.history[:, 'train_loss'], label='Training')
                                            ax.legend(handles=[p_vl, p_tl])
                                            plt.show()
                                            cp = skorch.callbacks.Checkpoint(dirname='results/dnn_checkpoints')
                                            mod_fnl.initialize()
                                            mod_fnl.load_params(checkpoint=cp)
                                            mod_fnl.fit(all_x, all_y)
                                        else:
                                            if mod_dict[k]['tune']:
                                                mod_fnl = mod_base.set_params(**mod_dict[k]['param_best'])
                                            else:
                                                mod_fnl = mod_base
            
                                            mod_fnl.fit(all_x, all_y)
                                        
                                        
                                
                                    if mod_dict[k]['bxfrm_y'] is not None:
                                        if mod_dict[k] == 'OLS':
                                            preds = mod_fnl.predict(df_test)
                                            preds[preds < 0] = 0
                                            preds = mod_dict[k]['bxfrm_y'](preds)
                                        else:
                                            preds = mod_fnl.predict(all_X).squeeze()
                                            preds[preds < 0] = 0
                                            preds = mod_dict[k]['bxfrm_y'](preds)
                                    else:
                                        if mod_dict[k] == 'OLS':
                                            preds = mod_fnl.predict(df_test)
                                            preds[preds < 0] = 0
                                        else:
                                            preds = mod_fnl.predict(all_X).squeeze()
                                            preds[preds < 0] = 0
                    
                                    # apply transformation to CPER 2022 model
                                    if (k == 'CPER_2022') and (cper_mod_xfrm):
                                        preds = cper_mod_xfrm_func(preds)
                                    
                                    #print('(time to fit: ' + str(round(time.time() - t0, 2)) + ' secs)')
                                
                                    mae_kg_tmp = np.nanmean(np.abs(preds - all_Y_orig)).round(3)
                                    mape_tmp = np.nanmean(np.abs(preds - all_Y_orig) / all_Y_orig).round(3)
                                    mae_pct_tmp = (mae_kg_tmp / np.nanmean(all_Y_orig)).round(3)
                                    r2_tmp = r2_score(all_Y_orig, preds).round(3)
                                    r_corr_tmp = r_corrcoef(all_Y_orig, preds).round(3)
                                    r2_tmp_xfrm = r2_score(mod_dict[k]['xfrm_y'](all_Y_orig),
                                                           mod_dict[k]['xfrm_y'](preds)).round(3)
                                    r_corr_tmp_xfrm = r_corrcoef(mod_dict[k]['xfrm_y'](all_Y_orig),
                                                                 mod_dict[k]['xfrm_y'](preds)).round(3)
        
                                    if agg_plot:
                                        df_test = df_sub.iloc[test_index].copy()
                                        df_test['preds'] = preds
                                        df_test['preds_xfrm'] = mod_dict[k]['xfrm_y'](preds)
                                        df_test[y_col + '_xfrm'] = df_test[y_col].apply(
                                            lambda x: mod_dict[k]['xfrm_y'](x))
                                        preds_plot = df_test.groupby([date_col, 'Plot'])['preds'].mean()
                                        all_Y_orig_plot = df_test.groupby([date_col, 'Plot'])[y_col].mean()
                                        preds_xfrm_plot = df_test.groupby([date_col, 'Plot'])['preds_xfrm'].mean()
                                        all_Y_orig_xfrm_plot = df_test.groupby([date_col, 'Plot'])[y_col + '_xfrm'].mean()
                                        mae_kg_tmp_plot = np.nanmean(np.abs(preds_plot - all_Y_orig_plot)).round(3)
                                        mape_tmp_plot = np.nanmean(np.abs(preds_plot - all_Y_orig_plot) / all_Y_orig_plot).round(3)
                                        mae_pct_tmp_plot = (mae_kg_tmp_plot / np.nanmean(all_Y_orig_plot)).round(3)
                                        r2_tmp_plot = r2_score(all_Y_orig_plot, preds_plot).round(3)
                                        r_corr_tmp_plot = r_corrcoef(all_Y_orig_plot, preds_plot).round(3)
                                        r2_tmp_xfrm_plot = r2_score(all_Y_orig_xfrm_plot, preds_xfrm_plot).round(3)
                                        r_corr_tmp_xfrm_plot = r_corrcoef(all_Y_orig_xfrm_plot, preds_xfrm_plot).round(3)
        
                                        df_results_tmp = pd.DataFrame({'Model': k,
                                                                       'numb_yrs': [yr_n - 1],
                                                                      'yr_train': [df_sub[date_col].dt.year.iloc[train_index].unique()],
                                                                      'yr_test': yr,
                                                                      'retune_bootstrap': retune_bootstrap,
                                                                      'MAE': mae_kg_tmp,
                                                                      'MAPE': mape_tmp,
                                                                      'MAE_pct': mae_pct_tmp,
                                                                      'R2': r2_tmp,
                                                                      'r_coef': r_corr_tmp,
                                                                      'R2_xfrm': r2_tmp_xfrm,
                                                                      'r_coef_xfrm': r_corr_tmp_xfrm,
                                                                       'MAE_plot': mae_kg_tmp_plot,
                                                                      'MAPE_plot': mape_tmp_plot,
                                                                      'MAE_pct_plot': mae_pct_tmp_plot,
                                                                      'R2_plot': r2_tmp_plot,
                                                                      'r_coef_plot': r_corr_tmp_plot,
                                                                      'R2_xfrm_plot': r2_tmp_xfrm_plot,
                                                                      'r_coef_xfrm_plot': r_corr_tmp_xfrm_plot},
                                                                     index=[idx_ct])
                                    else:
                                        df_results_tmp = pd.DataFrame({'Model': k,
                                                                              'numb_yrs': [yr_n - 1],
                                                                              'yr_train': [df_sub[date_col].dt.year.iloc[train_index].unique()],
                                                                              'yr_test': yr,
                                                                              'retune_bootstrap': retune_bootstrap,
                                                                              'MAE': mae_kg_tmp,
                                                                              'MAPE': mape_tmp,
                                                                              'MAE_pct': mae_pct_tmp,
                                                                              'R2': r2_tmp,
                                                                              'r_coef': r_corr_tmp},
                                                                             index=[idx_ct])
                                    if df_results_yrs is not None:
                                        df_results_yrs = pd.concat([df_results_yrs, df_results_tmp]).reset_index(drop=True)
                                    else:
                                        df_results_yrs = df_results_tmp.copy()
                                    idx_ct += 1
                                    if save_path is not None:
                                        df_results_yrs.to_csv(save_path, index=False)
                                    #if retune_bootstrap and (idx_ct%500==0):
                                    #    nworkers=len(client.ncores())
                                    #    client.restart(wait_for_workers=False)
                                    #    try:
                                    #        client.wait_for_workers(n_workers=int(nworkers*0.5), timeout=300)
                                    #    except dask.distributed.TimeoutError as e:
                                    #        print(str(int(nworkers*0.5)) + ' workers not available. Continuing with available workers.')
                                            #print(e)
                                    #        pass
                                    #    display(client)
                            else:
                                continue
    return df_results_yrs


def train_pred_ml_models_nyrs(nickname, mod_dict, df, y_col, date_col, var_names,
                               tuneby_group,
                               n_yrs_train,
                               backend, nthreads,
                               cper_mod_xfrm, cper_mod_xfrm_func, client,
                               cper_var_dict=cper_var_dict,
                               retune_bootstrap=True,
                               agg_plot=False,
                               save_path=None):
    from sklearn.metrics import r2_score
    import itertools
    from tqdm import tqdm

    if save_path is None:
        df_results_yrs = None
    else:
        if os.path.exists(save_path):
            df_results_yrs = pd.read_csv(save_path)
            check_iters=True
        else:
            df_results_yrs = None
            check_iters=False
            skip_combos=False
    
    mod_logo = LeaveOneGroupOut()
    mod_split = mod_logo

    scoring = {
        'MAE': 'neg_mean_absolute_error',
    }
    if backend == 'dask':
        from dask_ml.model_selection import GridSearchCV
        from sklearn.model_selection import GridSearchCV as skGridSearchCV
    else:
        from sklearn.model_selection import GridSearchCV
    
    idx_ct = 0
    for yr_n in [n_yrs_train]:
        print('Running ' + str(yr_n) + '-year combos')
        combos = list(itertools.combinations(df['Year'].unique(), yr_n))
        print(len(combos))
        if check_iters:
            if ((yr_n - 1) in df_results_yrs['numb_yrs'].unique()) and all(
                df_results_yrs[df_results_yrs['numb_yrs'] == yr_n - 1].groupby('Model').count()['numb_yrs']/yr_n == len(combos)):
                skip_combos=True
            else:
                skip_combos=False
        if skip_combos:
            print('All iterations run for all models. Skipping combos.')
            continue
        else:
            for yr_combo in tqdm(combos):
                df_sub = df[df['Year'].isin(yr_combo)]
                df_sub = df_sub.dropna(subset=var_names + [y_col])
                t0 = time.time()
                
                for train_index, test_index in mod_logo.split(df_sub, groups=df_sub['Year']):
                    yr = df_sub[date_col].dt.year.iloc[test_index].unique()[0]
                    #print(str(df_sub[date_col].dt.year.iloc[train_index].unique()))
                    #print(len(str(df_sub[date_col].dt.year.iloc[train_index].unique())))
                    #df_results_yrs = df_results_yrs.reset_index(drop=True)
                    train_yrs_tmp = re.sub(',',
                                           '', 
                                           str(df_sub[date_col].dt.year.iloc[train_index].unique()))
                    if df_results_yrs is not None:
                        outlen_tmp = sum(df_results_yrs['yr_train'].isin([train_yrs_tmp]) & (df_results_yrs['yr_test'] == yr))
                        if outlen_tmp == sum([int(mod_dict[i]['fit']) for i in mod_dict.keys()]):
                            skip_split = True
                        else:
                            skip_split = False
                    else:
                        skip_split = False
                    if skip_split:
                        continue
                    else:
                        logo_k = yr
                        train_loc = df_sub.iloc[train_index].index
                        test_loc = df_sub.iloc[test_index].index
                        
                        all_y_orig = df_sub[y_col].iloc[train_index]
                        all_Y_orig = df_sub[y_col].iloc[test_index]
                        all_x_orig = df_sub[var_names].iloc[train_index, :]
                        all_X_orig = df_sub[var_names].iloc[test_index, :]
                    
                        for k in mod_dict:
                            if mod_dict[k]['fit']:
                                #print('....fitting ' + k, end = " ")
                                run_check = k + ','.join(df_sub[date_col].dt.year.iloc[train_index].unique().astype(str))+str(yr)
                                if df_results_yrs is not None:
                                    run_check_result = run_check in df_results_yrs.apply(
                                        lambda x: x['Model']+x['yr_train']+str(x['yr_test']),
                                        axis=1).values
                                if df_results_yrs is not None and run_check_result:
                                        print('Skipping ' + run_check + '. Already saved.')
                                        continue
                                else:
                                    # prep data
                                    if mod_dict[k]['xfrm_y'] is not None:
                                        all_y = all_y_orig.apply(mod_dict[k]['xfrm_y'])     
                                        all_Y = all_Y_orig.apply(mod_dict[k]['xfrm_y'])
                                    else:
                                        all_y = all_y_orig.copy()
                                        all_Y = all_Y_orig.copy()
                                    if mod_dict[k]['scale_x']:
                                        scaler = mod_dict[k]['scaler']
                                        scaler.fit(all_x_orig)
                                        all_x = scaler.transform(all_x_orig)
                                        all_X = scaler.transform(all_X_orig)
                                    else:
                                        all_x = all_x_orig.copy()
                                        all_X = all_X_orig.copy()
                                    
                                    if mod_dict[k]['interactions']:
                                        poly_x = PolynomialFeatures(degree=mod_dict[k]['interaction_poly'], 
                                                                    interaction_only=mod_dict[k]['interaction_only'], include_bias = False)
                                        all_x = poly_x.fit_transform(all_x)
                                        poly_X = PolynomialFeatures(degree=mod_dict[k]['interaction_poly'], 
                                                                    interaction_only=mod_dict[k]['interaction_only'], include_bias = False)
                                        all_X = poly_X.fit_transform(all_X)
                                        var_names_out = poly_x.get_feature_names_out(var_names)
                                    else:
                                        var_names_out = var_names
                        
                                    # create a base model
                                    mod_base = mod_dict[k]['base_mod']
                                    # set parameters
                                    if retune_bootstrap:
                                        split_groups = df[tuneby_group].iloc[train_index]
                                        cv_splitter = mod_logo.split(all_x, groups=split_groups)
            
                                        if 'OLS' in k:
                                            df_train = pd.merge(pd.DataFrame(data=all_y),
                                                                pd.DataFrame(columns=all_x_orig.columns, data=all_x, index=all_x_orig.index),
                                                                left_index=True,
                                                                right_index=True)
                                            df_test = pd.merge(pd.DataFrame(data=all_Y),
                                                                pd.DataFrame(columns=all_X_orig.columns, data=all_X, index=all_X_orig.index),
                                                                left_index=True,
                                                                right_index=True)
                                            if k == 'OLS_2022':
                                                form_fnl = mod_dict[k]['base_mod']
                                            else:
                                                idx = 0
                                                df_results_list = []
                                                for k_fold in range(1, 3 + 1):
                                                    for combo in itertools.combinations(var_names, k_fold):
                                                        combo_corr = df[np.array(combo)].corr()
                                                        if ((combo_corr != 1.0) & (combo_corr.abs() > 0.8)).any(axis=None):
                                                            continue
                                                        else:
                                                            lr_form = mod_dict[k]['base_mod'] + combo[0]
                                                            if k_fold > 1:
                                                                for c in combo[1:]:
                                                                    lr_form = lr_form + ' + ' + c
                                                                for combo_c in itertools.combinations(combo, 2):
                                                                    lr_form = lr_form + ' + ' + combo_c[0] + ':' + combo_c[1]
                                                            df_results_tmp = fit_ols(all_x,
                                                                                     mod_split, 
                                                                                     split_groups,
                                                                                     df_train,
                                                                                     y_col,
                                                                                     lr_form, 
                                                                                     yr,
                                                                                     logo_k, 
                                                                                     k_fold,
                                                                                     idx)
                                                            df_results_list.append(df_results_tmp)
                                                            #mod_dict[k]['formula_df'] = pd.concat([df_results_tmp.compute(), mod_dict[k]['formula_df']])
                                                            #break
                                                df_results = dask.compute(df_results_list)
                                                mod_dict[k]['formula_df'] = pd.concat([mod_dict[k]['formula_df'], pd.concat(df_results[0])])
                                                if mod_dict[k]['tune_refit_type'] == 'minimize':
                                                    tune_loc = 0
                                                elif mod_dict[k]['tune_refit_type'] == 'maximize':
                                                    tune_loc = -1
                                                form_fnl = mod_dict[k]['formula_df'][mod_dict[k]['formula_df']['kfold'] == logo_k].sort_values(
                                                    mod_dict[k]['tune_refit'])['formula'].iloc[tune_loc]
                                            mod_fnl = smf.ols(formula=form_fnl, data=df_train).fit()
                                        
                                        elif k == 'MLP':
                                            from sklearn.model_selection import GridSearchCV as skGridSearchCV
                                            grid_search = skGridSearchCV(estimator=mod_base,
                                                                           param_grid=mod_dict[k]['param_grid'],
                                                                           scoring=scoring, 
                                                                           refit=mod_dict[k]['tune_refit'], 
                                                                           return_train_score=True,
                                                                           cv=cv_splitter, 
                                                                           n_jobs=min(sum([len(x) for x in mod_dict[k]['param_grid']]),
                                                                                      len(client.nthreads())))
                                            with parallel_backend('threading'):
                                                with warnings.catch_warnings():
                                                    warnings.simplefilter("ignore", category=ConvergenceWarning)
                                                    grid_search.fit(all_x, all_y)
                                            mod_fnl = mod_base.set_params(**grid_search.best_params_)
                                            mod_fnl.fit(all_x, all_y)
                                            mod_dict[k]['tune_results'][logo_k] = grid_search.cv_results_
                                        elif k == 'DNN':
                                            from sklearn.model_selection import GridSearchCV as skGridSearchCV
                                            grid_search = skGridSearchCV(estimator=mod_base,
                                                                         param_grid=mod_dict[k]['param_grid'],
                                                                         scoring=scoring, 
                                                                         refit=mod_dict[k]['tune_refit'], 
                                                                         return_train_score=True,
                                                                         cv=cv_splitter, 
                                                                         n_jobs=min(sum([len(x) for x in mod_dict[k]['param_grid']]),
                                                                                      len(client.nthreads())))
                                            grid_search.fit(all_x, all_y)
                                            mod_fnl = mod_base.set_params(**grid_search.best_params_)
                                            mod_fnl.fit(all_x, all_y)
                                            ax = plt.subplot()
                                            p_vl, = ax.plot(mod_fnl.history[:, 'valid_loss'], label='Validation')
                                            p_tl, = ax.plot(mod_fnl.history[:, 'train_loss'], label='Training')
                                            ax.legend(handles=[p_vl, p_tl])
                                            plt.show()
                                            mod_dict[k]['tune_results'][logo_k] = grid_search.cv_results_
                                        else:
                                            grid_search = GridSearchCV(estimator=mod_base,
                                                                           param_grid=mod_dict[k]['param_grid'],
                                                                           scoring=scoring, 
                                                                           refit=mod_dict[k]['tune_refit'], 
                                                                           return_train_score=True,
                                                                           cv=cv_splitter, 
                                                                           n_jobs=min(sum([len(x) for x in mod_dict[k]['param_grid']]),
                                                                                     nthreads))
                                            with parallel_backend(backend):
                                                with warnings.catch_warnings():
                                                    warnings.simplefilter("ignore", category=ConvergenceWarning)
                                                    grid_search.fit(all_x, all_y)
                                            mod_fnl = mod_base.set_params(**grid_search.best_params_)
                                            mod_fnl.fit(all_x, all_y)
                                            mod_dict[k]['tune_results'][logo_k] = grid_search.cv_results_
                                    else:
                                        if k == 'OLS_2022':
                                            form_fnl = mod_base
                                            df_train = pd.merge(pd.DataFrame(data=all_y),
                                                            pd.DataFrame(columns=all_x_orig.columns, data=all_x, index=all_x_orig.index),
                                                            left_index=True,
                                                            right_index=True)
                                            mod_fnl = smf.ols(formula=form_fnl, data=df_train).fit()
                                        elif k == 'OLS':
                                            form_fnl = mod_dict[k]['formula_df'].sort_values('mae_orig_mean')['formula'].iloc[0]
                                            df_train = pd.merge(pd.DataFrame(data=all_y),
                                                            pd.DataFrame(columns=all_x_orig.columns, data=all_x, index=all_x_orig.index),
                                                            left_index=True,
                                                            right_index=True)
                                            mod_fnl = smf.ols(formula=form_fnl, data=df_train).fit()
                                        elif k == 'DNN':
                                            mod_fnl = mod_base
                                            mod_fnl.fit(all_x, all_y)
                                            ax = plt.subplot()
                                            p_vl, = ax.plot(mod_fnl.history[:, 'valid_loss'], label='Validation')
                                            p_tl, = ax.plot(mod_fnl.history[:, 'train_loss'], label='Training')
                                            ax.legend(handles=[p_vl, p_tl])
                                            plt.show()
                                            cp = skorch.callbacks.Checkpoint(dirname='results/dnn_checkpoints')
                                            mod_fnl.initialize()
                                            mod_fnl.load_params(checkpoint=cp)
                                            mod_fnl.fit(all_x, all_y)
                                        else:
                                            if mod_dict[k]['tune']:
                                                mod_fnl = mod_base.set_params(**mod_dict[k]['param_best'])
                                            else:
                                                mod_fnl = mod_base
            
                                            mod_fnl.fit(all_x, all_y)
                                        
                                        
                                
                                    if mod_dict[k]['bxfrm_y'] is not None:
                                        if mod_dict[k] == 'OLS':
                                            preds = mod_fnl.predict(df_test)
                                            preds[preds < 0] = 0
                                            preds = mod_dict[k]['bxfrm_y'](preds)
                                        else:
                                            preds = mod_fnl.predict(all_X).squeeze()
                                            preds[preds < 0] = 0
                                            preds = mod_dict[k]['bxfrm_y'](preds)
                                    else:
                                        if mod_dict[k] == 'OLS':
                                            preds = mod_fnl.predict(df_test)
                                            preds[preds < 0] = 0
                                        else:
                                            preds = mod_fnl.predict(all_X).squeeze()
                                            preds[preds < 0] = 0
                    
                                    # apply transformation to CPER 2022 model
                                    if (k == 'CPER_2022') and (cper_mod_xfrm):
                                        preds = cper_mod_xfrm_func(preds)
                                    
                                
                                    df_results_tmp = pd.DataFrame({'Model': k,
                                                                   'numb_yrs': yr_n - 1,
                                                                   'yr_train': ','.join(df_sub[date_col].dt.year.iloc[train_index].unique().astype(str)),
                                                                   'yr_test': yr,
                                                                   'retune_bootstrap': retune_bootstrap,
                                                                   'predicted': preds})
                                    for c in [y_col, 'Id', 'Pasture', 'Date_mean', 'Year', 'Season']:
                                        df_results_tmp[c] = df_sub[c].iloc[test_index].values
                                    if df_results_yrs is not None:
                                        df_results_yrs = pd.concat([df_results_yrs, df_results_tmp]).reset_index(drop=True)
                                    else:
                                        df_results_yrs = df_results_tmp.copy()
                                    idx_ct += 1
                                    if save_path is not None:
                                        df_results_yrs.to_csv(save_path, index=False)
                                    #if retune_bootstrap and (idx_ct%500==0):
                                    #    nworkers=len(client.ncores())
                                    #    client.restart(wait_for_workers=False)
                                    #    try:
                                    #        client.wait_for_workers(n_workers=int(nworkers*0.5), timeout=300)
                                    #    except dask.distributed.TimeoutError as e:
                                    #        print(str(int(nworkers*0.5)) + ' workers not available. Continuing with available workers.')
                                            #print(e)
                                    #        pass
                                    #    display(client)
                            else:
                                continue
    return df_results_yrs


def train_pred_ml_models_nyrs_niters(nickname, mod_dict, df, y_col, date_col, var_names,
                                     tuneby_group,
                                     n_yrs_train,
                                     n_iters_train,
                               backend, nthreads,
                               cper_mod_xfrm, cper_mod_xfrm_func, client,
                               cper_var_dict=cper_var_dict,
                               retune_bootstrap=True,
                               agg_plot=False,
                               save_path=None):
    from sklearn.metrics import r2_score
    import itertools
    from tqdm import tqdm
    import random

    if save_path is None:
        df_results_yrs = None
    else:
        if os.path.exists(save_path):
            df_results_yrs = pd.read_csv(save_path)
            check_iters=True
        else:
            df_results_yrs = None
            check_iters=False
            skip_combos=False
    
    mod_logo = LeaveOneGroupOut()
    mod_split = mod_logo

    scoring = {
        'MAE': 'neg_mean_absolute_error',
    }
    if backend == 'dask':
        from dask_ml.model_selection import GridSearchCV
        from sklearn.model_selection import GridSearchCV as skGridSearchCV
    else:
        from sklearn.model_selection import GridSearchCV
    
    idx_ct = 0
    for yr_n in [n_yrs_train]:
        print('Running ' + str(yr_n) + '-year combos')
        combos = list(itertools.combinations(df['Year'].unique(), yr_n))
        print(len(combos))
        if check_iters:
            if ((yr_n - 1) in df_results_yrs['numb_yrs'].unique()) and all(
                df_results_yrs[df_results_yrs['numb_yrs'] == yr_n - 1].groupby('Model').count()['numb_yrs']/yr_n == len(combos)):
                skip_combos=True
            else:
                skip_combos=False
        if skip_combos:
            print('All iterations run for all models. Skipping combos.')
            continue
        else:
            for yr in tqdm(df['Year'].unique()):
                yr_combos = random.sample([x for x in combos if yr in x], n_iters_train)
                for yr_combo in tqdm(yr_combos):
                    df_sub = df[df['Year'].isin(yr_combo)]
                    df_sub = df_sub.dropna(subset=var_names + [y_col])
                    t0 = time.time()
                    test_index = df_sub.reset_index()[df_sub.reset_index()['Year'] == yr].index
                    train_index = df_sub.reset_index()[df_sub.reset_index()['Year'] != yr].index
                    train_yrs_tmp = re.sub(',',
                                           '', 
                                           str(df_sub[date_col].dt.year.iloc[train_index].unique()))
                    if df_results_yrs is not None:
                        outlen_tmp = sum(df_results_yrs['yr_train'].isin([train_yrs_tmp]) & (df_results_yrs['yr_test'] == yr))
                        if outlen_tmp == sum([int(mod_dict[i]['fit']) for i in mod_dict.keys()]):
                            skip_split = True
                        else:
                            skip_split = False
                    else:
                        skip_split = False
                    if skip_split:
                        continue
                    else:
                        logo_k = yr
                        train_loc = df_sub.iloc[train_index].index
                        test_loc = df_sub.iloc[test_index].index
                        
                        all_y_orig = df_sub[y_col].iloc[train_index]
                        all_Y_orig = df_sub[y_col].iloc[test_index]
                        all_x_orig = df_sub[var_names].iloc[train_index, :]
                        all_X_orig = df_sub[var_names].iloc[test_index, :]
                    
                        for k in mod_dict:
                            if mod_dict[k]['fit']:
                                #print('....fitting ' + k, end = " ")
                                run_check = k + ','.join(df_sub[date_col].dt.year.iloc[train_index].unique().astype(str))+str(yr)
                                if df_results_yrs is not None:
                                    run_check_result = run_check in df_results_yrs.apply(
                                        lambda x: x['Model']+x['yr_train']+str(x['yr_test']),
                                        axis=1).values
                                if df_results_yrs is not None and run_check_result:
                                        print('Skipping ' + run_check + '. Already saved.')
                                        continue
                                else:
                                    # prep data
                                    if mod_dict[k]['xfrm_y'] is not None:
                                        all_y = all_y_orig.apply(mod_dict[k]['xfrm_y'])     
                                        all_Y = all_Y_orig.apply(mod_dict[k]['xfrm_y'])
                                    else:
                                        all_y = all_y_orig.copy()
                                        all_Y = all_Y_orig.copy()
                                    if mod_dict[k]['scale_x']:
                                        scaler = mod_dict[k]['scaler']
                                        scaler.fit(all_x_orig)
                                        all_x = scaler.transform(all_x_orig)
                                        all_X = scaler.transform(all_X_orig)
                                    else:
                                        all_x = all_x_orig.copy()
                                        all_X = all_X_orig.copy()
                                    
                                    if mod_dict[k]['interactions']:
                                        poly_x = PolynomialFeatures(degree=mod_dict[k]['interaction_poly'], 
                                                                    interaction_only=mod_dict[k]['interaction_only'], include_bias = False)
                                        all_x = poly_x.fit_transform(all_x)
                                        poly_X = PolynomialFeatures(degree=mod_dict[k]['interaction_poly'], 
                                                                    interaction_only=mod_dict[k]['interaction_only'], include_bias = False)
                                        all_X = poly_X.fit_transform(all_X)
                                        var_names_out = poly_x.get_feature_names_out(var_names)
                                    else:
                                        var_names_out = var_names
                        
                                    # create a base model
                                    mod_base = mod_dict[k]['base_mod']
                                    # set parameters
                                    if retune_bootstrap:
                                        split_groups = df[tuneby_group].iloc[train_index]
                                        cv_splitter = mod_logo.split(all_x, groups=split_groups)
            
                                        if 'OLS' in k:
                                            df_train = pd.merge(pd.DataFrame(data=all_y),
                                                                pd.DataFrame(columns=all_x_orig.columns, data=all_x, index=all_x_orig.index),
                                                                left_index=True,
                                                                right_index=True)
                                            df_test = pd.merge(pd.DataFrame(data=all_Y),
                                                                pd.DataFrame(columns=all_X_orig.columns, data=all_X, index=all_X_orig.index),
                                                                left_index=True,
                                                                right_index=True)
                                            if k == 'OLS_2022':
                                                form_fnl = mod_dict[k]['base_mod']
                                            else:
                                                idx = 0
                                                df_results_list = []
                                                for k_fold in range(1, 3 + 1):
                                                    for combo in itertools.combinations(var_names, k_fold):
                                                        combo_corr = df[np.array(combo)].corr()
                                                        if ((combo_corr != 1.0) & (combo_corr.abs() > 0.8)).any(axis=None):
                                                            continue
                                                        else:
                                                            lr_form = mod_dict[k]['base_mod'] + combo[0]
                                                            if k_fold > 1:
                                                                for c in combo[1:]:
                                                                    lr_form = lr_form + ' + ' + c
                                                                for combo_c in itertools.combinations(combo, 2):
                                                                    lr_form = lr_form + ' + ' + combo_c[0] + ':' + combo_c[1]
                                                            df_results_tmp = fit_ols(all_x,
                                                                                     mod_split, 
                                                                                     split_groups,
                                                                                     df_train,
                                                                                     y_col,
                                                                                     lr_form, 
                                                                                     yr,
                                                                                     logo_k, 
                                                                                     k_fold,
                                                                                     idx)
                                                            df_results_list.append(df_results_tmp)
                                                            #mod_dict[k]['formula_df'] = pd.concat([df_results_tmp.compute(), mod_dict[k]['formula_df']])
                                                            #break
                                                df_results = dask.compute(df_results_list)
                                                mod_dict[k]['formula_df'] = pd.concat([mod_dict[k]['formula_df'], pd.concat(df_results[0])])
                                                if mod_dict[k]['tune_refit_type'] == 'minimize':
                                                    tune_loc = 0
                                                elif mod_dict[k]['tune_refit_type'] == 'maximize':
                                                    tune_loc = -1
                                                form_fnl = mod_dict[k]['formula_df'][mod_dict[k]['formula_df']['kfold'] == logo_k].sort_values(
                                                    mod_dict[k]['tune_refit'])['formula'].iloc[tune_loc]
                                            mod_fnl = smf.ols(formula=form_fnl, data=df_train).fit()
                                        
                                        elif k == 'MLP':
                                            from sklearn.model_selection import GridSearchCV as skGridSearchCV
                                            grid_search = skGridSearchCV(estimator=mod_base,
                                                                           param_grid=mod_dict[k]['param_grid'],
                                                                           scoring=scoring, 
                                                                           refit=mod_dict[k]['tune_refit'], 
                                                                           return_train_score=True,
                                                                           cv=cv_splitter, 
                                                                           n_jobs=min(sum([len(x) for x in mod_dict[k]['param_grid']]),
                                                                                      len(client.nthreads())))
                                            with parallel_backend('threading'):
                                                with warnings.catch_warnings():
                                                    warnings.simplefilter("ignore", category=ConvergenceWarning)
                                                    grid_search.fit(all_x, all_y)
                                            mod_fnl = mod_base.set_params(**grid_search.best_params_)
                                            mod_fnl.fit(all_x, all_y)
                                            mod_dict[k]['tune_results'][logo_k] = grid_search.cv_results_
                                        elif k == 'DNN':
                                            from sklearn.model_selection import GridSearchCV as skGridSearchCV
                                            grid_search = skGridSearchCV(estimator=mod_base,
                                                                         param_grid=mod_dict[k]['param_grid'],
                                                                         scoring=scoring, 
                                                                         refit=mod_dict[k]['tune_refit'], 
                                                                         return_train_score=True,
                                                                         cv=cv_splitter, 
                                                                         n_jobs=min(sum([len(x) for x in mod_dict[k]['param_grid']]),
                                                                                      len(client.nthreads())))
                                            grid_search.fit(all_x, all_y)
                                            mod_fnl = mod_base.set_params(**grid_search.best_params_)
                                            mod_fnl.fit(all_x, all_y)
                                            ax = plt.subplot()
                                            p_vl, = ax.plot(mod_fnl.history[:, 'valid_loss'], label='Validation')
                                            p_tl, = ax.plot(mod_fnl.history[:, 'train_loss'], label='Training')
                                            ax.legend(handles=[p_vl, p_tl])
                                            plt.show()
                                            mod_dict[k]['tune_results'][logo_k] = grid_search.cv_results_
                                        else:
                                            grid_search = GridSearchCV(estimator=mod_base,
                                                                           param_grid=mod_dict[k]['param_grid'],
                                                                           scoring=scoring, 
                                                                           refit=mod_dict[k]['tune_refit'], 
                                                                           return_train_score=True,
                                                                           cv=cv_splitter, 
                                                                           n_jobs=min(sum([len(x) for x in mod_dict[k]['param_grid']]),
                                                                                     nthreads))
                                            with parallel_backend(backend):
                                                with warnings.catch_warnings():
                                                    warnings.simplefilter("ignore", category=ConvergenceWarning)
                                                    grid_search.fit(all_x, all_y)
                                            mod_fnl = mod_base.set_params(**grid_search.best_params_)
                                            mod_fnl.fit(all_x, all_y)
                                            mod_dict[k]['tune_results'][logo_k] = grid_search.cv_results_
                                    else:
                                        if k == 'OLS_2022':
                                            form_fnl = mod_base
                                            df_train = pd.merge(pd.DataFrame(data=all_y),
                                                            pd.DataFrame(columns=all_x_orig.columns, data=all_x, index=all_x_orig.index),
                                                            left_index=True,
                                                            right_index=True)
                                            mod_fnl = smf.ols(formula=form_fnl, data=df_train).fit()
                                        elif k == 'OLS':
                                            form_fnl = mod_dict[k]['formula_df'].sort_values('mae_orig_mean')['formula'].iloc[0]
                                            df_train = pd.merge(pd.DataFrame(data=all_y),
                                                            pd.DataFrame(columns=all_x_orig.columns, data=all_x, index=all_x_orig.index),
                                                            left_index=True,
                                                            right_index=True)
                                            mod_fnl = smf.ols(formula=form_fnl, data=df_train).fit()
                                        elif k == 'DNN':
                                            mod_fnl = mod_base
                                            mod_fnl.fit(all_x, all_y)
                                            ax = plt.subplot()
                                            p_vl, = ax.plot(mod_fnl.history[:, 'valid_loss'], label='Validation')
                                            p_tl, = ax.plot(mod_fnl.history[:, 'train_loss'], label='Training')
                                            ax.legend(handles=[p_vl, p_tl])
                                            plt.show()
                                            cp = skorch.callbacks.Checkpoint(dirname='results/dnn_checkpoints')
                                            mod_fnl.initialize()
                                            mod_fnl.load_params(checkpoint=cp)
                                            mod_fnl.fit(all_x, all_y)
                                        else:
                                            if mod_dict[k]['tune']:
                                                mod_fnl = mod_base.set_params(**mod_dict[k]['param_best'])
                                            else:
                                                mod_fnl = mod_base
            
                                            mod_fnl.fit(all_x, all_y)
                                        
                                        
                                
                                    if mod_dict[k]['bxfrm_y'] is not None:
                                        if mod_dict[k] == 'OLS':
                                            preds = mod_fnl.predict(df_test)
                                            preds[preds < 0] = 0
                                            preds = mod_dict[k]['bxfrm_y'](preds)
                                        else:
                                            preds = mod_fnl.predict(all_X).squeeze()
                                            preds[preds < 0] = 0
                                            preds = mod_dict[k]['bxfrm_y'](preds)
                                    else:
                                        if mod_dict[k] == 'OLS':
                                            preds = mod_fnl.predict(df_test)
                                            preds[preds < 0] = 0
                                        else:
                                            preds = mod_fnl.predict(all_X).squeeze()
                                            preds[preds < 0] = 0
                    
                                    # apply transformation to CPER 2022 model
                                    if (k == 'CPER_2022') and (cper_mod_xfrm):
                                        preds = cper_mod_xfrm_func(preds)
                                    
                                
                                    df_results_tmp = pd.DataFrame({'Model': k,
                                                                   'numb_yrs': yr_n - 1,
                                                                   'yr_train': ','.join(df_sub[date_col].dt.year.iloc[train_index].unique().astype(str)),
                                                                   'yr_test': yr,
                                                                   'retune_bootstrap': retune_bootstrap,
                                                                   'predicted': preds})
                                    for c in [y_col, 'Id', 'Pasture', 'Date_mean', 'Year', 'Season']:
                                        df_results_tmp[c] = df_sub[c].iloc[test_index].values
                                    if df_results_yrs is not None:
                                        df_results_yrs = pd.concat([df_results_yrs, df_results_tmp]).reset_index(drop=True)
                                    else:
                                        df_results_yrs = df_results_tmp.copy()
                                    idx_ct += 1
                                    if save_path is not None:
                                        df_results_yrs.to_csv(save_path, index=False)
                                    #if retune_bootstrap and (idx_ct%500==0):
                                    #    nworkers=len(client.ncores())
                                    #    client.restart(wait_for_workers=False)
                                    #    try:
                                    #        client.wait_for_workers(n_workers=int(nworkers*0.5), timeout=300)
                                    #    except dask.distributed.TimeoutError as e:
                                    #        print(str(int(nworkers*0.5)) + ' workers not available. Continuing with available workers.')
                                            #print(e)
                                    #        pass
                                    #    display(client)
                            else:
                                continue
    return df_results_yrs