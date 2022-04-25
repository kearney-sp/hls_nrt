import os
import pandas as pd
import pickle
import param as pm
import panel as pn
import numpy as np
import xarray as xr
from cartopy import crs
import geoviews as gv
import holoviews as hv
import hvplot.xarray
from copy import deepcopy
from holoviews import streams
import affine
from skimage.draw import polygon
from bokeh.models.formatters import PrintfTickFormatter
from bokeh.models import NumeralTickFormatter
from datetime import datetime, timedelta
from pyproj import Proj
from src.hls_funcs import fetch
from src.hls_funcs.masks import mask_hls, shp2mask, bolton_mask
from src.hls_funcs.indices import ndvi_func
from src.hls_funcs.smooth import smooth_xr, despike_ts_xr
from src.hls_funcs.predict import pred_bm, pred_cov, pred_bm_se, pred_bm_thresh, xr_cdf

pn.extension()
pn.param.ParamMethod.loading_indicator = True

cper_f = os.path.join('data/ground/cper_pastures_2017_clip.shp')
mod_bm = pickle.load(open('src/models/CPER_HLS_to_VOR_biomass_model_lr_simp.pk', 'rb'))

print('   setting up Local cluster...')
from dask.distributed import LocalCluster, Client
import dask
aws=False
fetch.setup_env(aws=aws)
cluster = LocalCluster(n_workers=1, threads_per_worker=2)
client = Client(cluster)

class getData(pm.Parameterized):
    bm_mod = pickle.load(open('src/models/CPER_HLS_to_VOR_biomass_model_lr_simp.pk', 'rb'))
    cov_mod = pickle.load(open('src/models/CPER_HLS_to_LPI_cover_pls_binned_model.pk', 'rb'))
    action = pm.Action(lambda x: x.param.trigger('action'), label='Load Data and Run Analysis')
    get_stats = pm.Action(lambda y: y.param.trigger('get_stats'), label='Calculate static stats')
    get_stats_ts = pm.Action(lambda y: y.param.trigger('get_stats_ts'), label='Calculate time-series stats')
    aws_sel = pn.widgets.Checkbox(name='Use AWS', value=False)
    username_input = pn.widgets.PasswordInput(name='NASA Earthdata Login', placeholder='Enter Username...')
    password_input = pn.widgets.PasswordInput(name='', placeholder='Enter Password...')
    year_picker = pn.widgets.Select(name='Year', options=[2021], value=2021)
    thresh_picker = pn.widgets.IntSlider(name='Threshold', start=200, end=2000, step=25, value=500,
                                 format=PrintfTickFormatter(format='%d kg/ha'))
    
    data_dict = {'date_range': [str(datetime(year_picker.value, 1, 1).date()),
                                str(datetime(year_picker.value, 12, 31).date())]}
 
    chunks = {'date': 1, 'y': 250, 'x': 250}
    datCRS = crs.UTM(13)
    mapCRS = crs.GOOGLE_MERCATOR
    datProj = Proj(datCRS.proj4_init)
    mapProj = Proj(mapCRS.proj4_init)
    map_args = dict(crs=datCRS, rasterize=False, project=True, dynamic=True)
    base_opts = dict(projection=mapCRS, backend='bokeh', xticks=None, yticks=None, width=900, height=700,
                         padding=0, active_tools=['pan', 'wheel_zoom'], toolbar='left')
    map_opts = dict(projection=mapCRS, responsive=False, xticks=None, yticks=None, width=900, height=700,
                     padding=0, tools=['pan', 'wheel_zoom', 'box_zoom'],
                     active_tools=['pan', 'wheel_zoom'], toolbar='left')
    
    poly_opts = dict(fill_color=['', ''], fill_alpha=[0.0, 0.0], line_color=['#1b9e77', '#d95f02'],
                 line_width=[3, 3])    
    bg_col='#ffffff'

    css = '''
    .bk.box1 {
      background: #ffffff;
      border-radius: 5px;
      border: 1px black solid;
    }
    '''
    
    date = pn.widgets.DatePicker(name='Calendar', width=200)
      
    box_ctr_dat = [522435.0, 4519550.5]
    box_ctr_coord = datProj(box_ctr_dat[0], box_ctr_dat[1], inverse=True)
    box_ctr_map = mapProj(box_ctr_coord[0], box_ctr_coord[1], inverse=False)
    
    tiles = gv.tile_sources.EsriImagery.opts(**base_opts, level='glyph',
                                                                   xlim=(box_ctr_map[0] - 5000,
                                                                         box_ctr_map[0] + 5000),
                                                                   ylim=(box_ctr_map[1] - 5000,
                                                                         box_ctr_map[1] + 5000))
    labels = gv.tile_sources.EsriReference.opts(**base_opts, level='overlay')
       
    basemap = gv.tile_sources.EsriImagery.opts(projection=mapCRS, backend='bokeh', level='glyph')
    
    base_rng = hv.streams.RangeXY(source=tiles)
    
    polys = hv.Polygons([])

    poly_stream = streams.PolyDraw(source=polys, drag=True, num_objects=2,
                                    show_vertices=True, styles=poly_opts)
    edit_stream = streams.PolyEdit(source=polys, shared=True)
    
    gauge_obj = pn.indicators.Gauge(
        name='Biomass', bounds=(0, 2500), format='{value} kg/ha',
        colors=[(0.20, '#FF6E76'), (0.40, '#FDDD60'), (0.60, '#7CFFB2'), (1, '#58D9F9')],
        num_splits=5, value=0, align='center', title_size=12,
        start_angle=180, end_angle=0, height=200, width=300,
    )
    
    def __init__(self, **params):
        super(getData, self).__init__(**params)
        self.da = None
        self.da_sel = None
        self.da_cov = None
        self.da_bm = None
        self.da_bm_se = None
        self.da_thresh = None
        self.all_maps = None
        self.cov_map = None
        self.bm_map = None
        self.thresh_map = None
        
        self.cov_stats = None
        self.bm_stats = None
        self.thresh_stats = None
        self.stats_titles = None
                
        self.view = self._create_view()
    
    @pm.depends('action')
    def proc_data(self):
        message = 'Not yet launched'
        if self.username_input.value != '':
            try:
                fetch.setup_env(aws=self.aws_sel.value, creds=[self.username_input.value,
                                                 self.password_input.value])
                coord_rng = self.mapProj(self.base_rng.x_range, self.base_rng.y_range, inverse=True)
                bounds_latlon = [coord_rng[0][0], coord_rng[1][0], coord_rng[0][1], coord_rng[1][1]]
                self.da = fetch.get_hls(hls_data=self.data_dict, 
                                         bbox_latlon=bounds_latlon, 
                                         aws=self.aws_sel.value)
                message = self.base_rng.x_range
                message = coord_rng
                dat_rng = np.round(self.datProj(coord_rng[0], coord_rng[1], inverse=False), 0)
                message = dat_rng
                #self.da = tmp_data.loc[dict(x=slice(*dat_rng[0]), y=slice(*dat_rng[1][::-1]))]
                self.da['time'] = self.da.time.dt.floor("D")
                self.da = self.da.rename(dict(time='date'))
                self.da = self.da.chunk(self.chunks)
                da_mask = mask_hls(self.da['FMASK'])
                self.da = self.da.where(da_mask == 0)
                self.da = self.da.sel(date=self.da['eo:cloud_cover'] < 80)
                self.date.enabled_dates = [pd.Timestamp(x).to_pydatetime().date() for x in self.da['date'].values]
                if self.date.value is None:
                    self.da_sel = self.da.isel(date=-1).compute()
                    self.date.value = pd.to_datetime(self.da_sel.date.values).date()
                else:
                    date_idx = self.date.enabled_dates.index(self.date.value)
                    self.da_sel = self.da.isel(date=date_idx).compute()
                message = 'Success!' + str(datetime.now())
                return message
            except:
                return message + ': App Failure'
        else:
            return message
      
    @pm.depends('date.param')
    def create_maps(self):
        if self.da is not None and self.da_sel is not None:
            if pd.to_datetime(self.da_sel.date.values).date() != self.date.value:
                if self.date.value is not None:
                    date_idx = self.date.enabled_dates.index(self.date.value)
                    self.da_sel = self.da.isel(date=date_idx).compute()
            if self.edit_stream.data is not None:
                self.polys = self.edit_stream.element
                self.poly_stream = streams.PolyDraw(source=self.polys, drag=True, num_objects=2,
                                                    show_vertices=True, styles=self.poly_opts)
                self.edit_stream = streams.PolyEdit(source=self.polys, shared=True)
            elif self.poly_stream.data is not None:
                self.polys = self.poly_stream.element
                self.poly_stream = streams.PolyDraw(source=self.polys, drag=True, num_objects=2,
                               show_vertices=True, styles=self.poly_opts)
                self.edit_stream = streams.PolyEdit(source=self.polys, shared=True)
            else:
                self.polys = self.polys
            da_bm = self.da_sel.map_blocks(pred_bm, template=self.da_sel['BLUE'],
                                     kwargs=dict(model=self.bm_mod))
            da_bm = da_bm.where(da_bm > 0)
            da_bm.name = 'Biomass'
            
            da_cov_temp = self.da_sel[['BLUE', 'GREEN', 'RED', 'NIR1']].rename(dict(BLUE='SD', RED='BARE', NIR1='LITT'))
            da_cov = self.da_sel.map_blocks(pred_cov, template=da_cov_temp,
                         kwargs=dict(model=self.cov_mod))
            da_cov = da_cov[['SD', 'GREEN', 'BARE']].to_array(dim='type')
            da_cov = da_cov.where((da_cov < 1.0) | (da_cov.isnull()), 1.0)
            da_cov.name = 'Cover'
            
            da_bm_se = self.da_sel.map_blocks(pred_bm_se, template=self.da_sel['BLUE'],
                                         kwargs=dict(model=self.bm_mod))
           
            da_thresh_pre = (np.log(self.thresh_picker.value) - xr.ufuncs.log(da_bm)) / da_bm_se
            da_thresh = da_thresh_pre.map_blocks(xr_cdf, template=self.da_sel['BLUE'])
            #da_thresh = pred_bm_thresh(da_bm, da_bm_se, self.thresh_picker.value)
            da_thresh.name = 'Threshold'
            
            da_ndvi = ndvi_func(self.da_sel)
            
            self.cov_map = da_cov.hvplot.rgb(x='x', y='y', bands='type',
                                        **self.map_args).opts(**self.map_opts)
            self.bm_map = da_bm.hvplot(x='x', y='y',
                                  cmap='Inferno', clim=(100, 1000), colorbar=False,
                                  **self.map_args).opts(**self.map_opts)
            self.thresh_map = da_thresh.hvplot(x='x', y='y',
                                          cmap='YlOrRd', clim=(0.05, 0.95), colorbar=False,
                                          **self.map_args).opts(**self.map_opts)
            
            self.ndvi_map = da_ndvi.hvplot(x='x', y='y',
                              cmap='Viridis', clim=(0.05, 0.50), colorbar=False,
                              **self.map_args).opts(**self.map_opts)

            self.da_cov = da_cov
            self.da_bm = da_bm
            self.da_bm_se = da_bm_se
            self.da_thresh = da_thresh
            self.da_ndvi = da_ndvi
            
            self.all_maps = pn.Tabs(
                ('Cover', self.basemap * self.cov_map * self.polys),
                ('Biomass', self.basemap * self.bm_map * self.polys),
                ('Biomass threshold', self.basemap * self.thresh_map * self.polys), 
                ('Greenness (NDVI)', self.basemap * self.ndvi_map * self.polys))
            
            #self.poly_stream = streams.PolyDraw(source=self.polys, drag=True, num_objects=2,
            #                        show_vertices=True, styles=self.poly_opts)
        
            return self.all_maps
        else:
            return pn.Column(self.tiles * self.labels)
        
    @pm.depends('get_stats')
    def show_stats(self):
        if self.poly_stream.data is not None:
            markdown_list = []
            thresh_list = []
            bm_list = []
            cov_list = []
            stats_rows = []
            for idx, ps_c in enumerate(self.poly_stream.data['line_color']):
                r, c = polygon(self.poly_stream.data['xs'][idx]*-1, self.poly_stream.data['ys'][idx])
                r = r * -1.0
                
                da_cov_tmp = self.cov_map[r][:,c].data
                cov_factors = [k for k in app.cov_map.data.keys() if k not in ['y', 'x']]
                cov_labels = [cov_dict[i] for i in cov_factors]
                cov_labels.append('Litter')
                cov_vals = [round(float(da_cov_tmp[f].mean()), 2) for f in cov_factors]
                cov_vals.append(round(1.0 - np.sum(cov_vals), 2))
                pct_fmt = NumeralTickFormatter(format="0%")
                cov_colors = hv.Cycle(['red', 'green', 'blue', 'orange'])
                cov_scatter_tmp = hv.Overlay([hv.Scatter(f) for f in list(zip(cov_labels, cov_vals))]) \
                    .options({'Scatter': dict(xformatter=pct_fmt,
                                              size=15,
                                              fill_color=cov_colors,
                                              line_color=cov_colors,
                                              ylim=(0, 1))})
                cov_spike_tmp = hv.Overlay([hv.Spikes(f) for f in cov_scatter_tmp])\
                    .options({'Spikes': dict(color=cov_colors, line_width=4,
                                             labelled=[], invert_axes=True, color_index=None,
                                             ylim=(0, 1))})
                cov_list.append(pn.Column((cov_spike_tmp * cov_scatter_tmp).options(height=200,
                                                                                    width=300,
                                                                                    bgcolor=self.bg_col,
                                                                                    toolbar=None),
                                          css_classes=['box1'], margin=5, align='center'))
                bm_dat_tmp = self.bm_map[r][:,c].data
                bm_hist_tmp = bm_dat_tmp.hvplot.hist('Biomass', xlim=(0, 2000),
                                                    bins=np.arange(0, 10000, 20))\
                    .opts(height=200, width=300, fill_color=ps_c, fill_alpha=0.6,
                          line_color='black', line_width=0.5, line_alpha=0.6,
                          bgcolor=self.bg_col, align='center').options(toolbar=None)
                bm_gauge_obj = deepcopy(self.gauge_obj)
                bm_gauge_obj.value = int(bm_dat_tmp.mean()['Biomass'])
                bm_list.append(pn.Column(bm_gauge_obj,
                                         css_classes=['box1'], margin=5, align='center'))
                markdown = pn.pane.Markdown('# Region ' + str(idx+1) + ' :', align='center',
                                            style={'font-family': "serif",
                                                   'color': ps_c})
                markdown_list.append(pn.Column(markdown, css_classes=['box1'], margin=5, align='center'))
                thresh_pct = round(float(bm_dat_tmp.where(bm_dat_tmp < 500).count()['Biomass'])/
                                   float(bm_dat_tmp.count()['Biomass']) * 100, 0)
                thresh_text = pn.pane.Markdown(f'**{thresh_pct}%** of the region is estimated to have biomass ' +
                                               f'less than {500} kg/ha.',
                                               style={'font-family': "Helvetica"})
                thresh_list.append(pn.Column(bm_hist_tmp * hv.VLine(x=500).opts(line_color='black', height=200),
                                             thresh_text,
                                             css_classes=['box1'], margin=5, align='center'))
                stats_rows.append(pn.Row(markdown_list[idx], bm_list[idx], thresh_list[idx], cov_list[idx]))
            self.stats_titles = pn.Column(*markdown_list)
            self.cov_stats = pn.Column(*cov_list)
            self.bm_stats = pn.Column(*bm_list)
            self.thresh_stats = pn.Column(*thresh_list)
            return pn.Column(*stats_rows)
        else:
            return pn.Row(None)
    
    def _create_view(self):
        layout = pn.Column(pn.Row(pn.Column(self.aws_sel, 
                                            self.username_input, self.password_input,
                                            pn.Param(self.param,
                                                     widgets={'action': pn.widgets.Button(name='Get data', width=200),
                                                              'get_stats': pn.widgets.Button(name='Calculate static stats', width=200),
                                                              'get_stats_ts': pn.widgets.Button(name='Calculate time-series stats', width=200)},
                                                     show_name=False),
                                            self.proc_data,
                                            self.date), pn.Column(self.create_maps)),
                           self.show_stats)
        return layout
                        
app = getData()
app.view.servable()
