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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, LinearRegression, ElasticNet
from sklearn.pipeline import Pipeline
import torch
import torch.nn as nn
import torch.optim as optim
import skorch
from skorch import NeuralNetRegressor

import sys
sys.path.insert(1, '/project/cper_neon_aop/hls_nrt/extract')
sys.path.insert(1, '/project/cper_neon_aop/hls_nrt/fit')
from dnn_setup import ResNetRegressor, ResidualBlock

import warnings
from sklearn.exceptions import ConvergenceWarning

lr_mod = pickle.load(open("/project/cper_neon_aop/cper_hls_veg_models/models/biomass/CPER_HLS_to_VOR_biomass_model_lr_simp.pk", 'rb'))

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
            'log_y': True,
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
            'log_y': True,
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
            'log_y': True,
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
            'log_y': True,
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
            'log_y': True,
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
            'log_y': True,
            'interactions': False,
            'interaction_only': True,
            'interaction_poly': 2,
            'variable_df': pd.DataFrame(columns=['kfold', 'Variable'] + ['PC_' + str(i+1) for i in range(len(var_names))])
        },
        'SVR': {
            'base_mod': Pipeline(
                [
                    ('scaler', StandardScaler()), 
                    ('SVR', SVR(kernel='linear'))
                ]),
            'fit': True,
            'variable_importance': True,
            'tune': True,
            'tune_refit': 'MAE',
            'param_grid': {
                'SVR__C': np.logspace(1.5, 4, 10, base=10),
                'SVR__gamma': np.logspace(-3.5, 0, 10, base=10)
            },
            'tune_results': {},
            'scale_x': False,
            'scaler': StandardScaler(),
            'log_y': False,
            'interactions': False,
            'interaction_only': True,
            'interaction_poly': 2,
            'variable_df': pd.DataFrame(columns=['kfold', 'Variable', 'SVR_weights'])
        },
        'RF': {
            'base_mod': Pipeline(
                [
                    ('scaler', StandardScaler()), 
                    ('RF', RandomForestRegressor(n_estimators=200, bootstrap=True, oob_score=True, n_jobs=-1))
                ]),
            'fit': True,
            'tune': True,
            'variable_importance': True,
            'tune_refit': 'MAE',
            'param_grid': {
                'RF__min_samples_split': [0.0001, 0.001, 0.005, 0.01],
                #'n_estimators': [400],
                'RF__max_samples': [0.2, 0.3, 0.5, 0.7, 0.9],
                'RF__max_features': [0.1, 0.25, 0.5, 0.75, 1.0]
            },
            'tune_results': {},
            'scale_x': False,
            'scaler': StandardScaler(),
            'log_y': False,
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
            'fit': True,
            'variable_importance': True,
            'tune': True,
            'tune_refit': 'MAE',
            'param_grid': {
                'GBR__learning_rate': [0.01, 0.025, 0.05, 0.075, 0.1],
                'GBR__min_samples_split': [0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
                'GBR__n_estimators': [100, 200, 400, 600, 800],
                'GBR__max_features': [0.1, 0.25, 0.5]
            },
            'tune_results': {},
            'scale_x': False,
            'scaler': StandardScaler(),
            'log_y': False,
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
            'log_y': False,
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
                'optimizer__momentum': [0.75, 0.85, 0.95],
                'optimizer__weight_decay': [1e-2, 1e-4],
                'optimizer__nesterov': [True],
                'lr': [0.01, 0.001, 0.0001],
                'batch_size': [64],
                'max_epochs': [100],
            },
            'tune_results': {},
            'scale_x': True,
            'scaler': MinMaxScaler(),
            'log_y': False,
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
                  cper_mod_xfrm, cper_mod_xfrm_func, cper_var_dict=cper_var_dict):
    if os.path.exists(outFILE_tmp):
        print('Output file already exists. Loading saved dataset.')
        df = pd.read_csv(outFILE_tmp, parse_dates=[date_col])
        with open(os.path.join(outDIR, 'tmp', 'ml_train_' + nickname + '_cv_' + kfold_group + '_tuneby_' + tuneby_group + '_results.pk'), 'rb') as f:
            mod_dict = pickle.load(f)
    else:
        for k in mod_dict:
            if mod_dict[k]['fit']:
                df[k] = np.nan
    
    mod_logo = LeaveOneGroupOut()
    mod_groupk = GroupKFold(n_splits=10)
    
    if kfold_type == 'logo':
        mod_split = mod_logo
    elif kfold_type == 'group_k':
        mod_split = mod_groupk
        kfold = 0
        df['kfold'] = np.nan
        
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
    
    restart_dask = False
    
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
                    #restart_dask = True
                    print('....fitting ' + k, end = " ")
                    t0 = time.time()
                    
                    # prep data
                    if mod_dict[k]['log_y']:
                        all_y = np.log(1 + all_y_orig)
                        all_Y = np.log(1 + all_Y_orig)
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
                        #if k in ['SVR', 'RF', 'GBR']:
                        #    rf_pi = permutation_importance(mod_fnl, all_X, all_Y, n_repeats=10, n_jobs=-1)
                        #    mod_dict[k]['variable_df'] = pd.concat([mod_dict[k]['variable_df'],
                        #                                            pd.DataFrame({'kfold': logo_k,
                        #                                                          'Variable': var_names_out,
                        #                                                          'PI': rf_pi.importances_mean})])
                    
                    if mod_dict[k]['log_y']:
                        if mod_dict[k] == 'OLS':
                            preds = np.exp(mod_fnl.predict(df_train))
                        else:
                            preds = np.exp(mod_fnl.predict(all_X).squeeze()) + 1
                    else:
                        if mod_dict[k] == 'OLS':
                            preds = mod_fnl.predict(df_train)
                        else:
                            preds = mod_fnl.predict(all_X).squeeze()
    
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