import glob
import h5py
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from astropy.time import Time

def N(x):
    """ Convert a string to a number if possible """
    try:
        return float(x)
    except:
        try:
            return float(x.replace('<',''))
        except:
            return x

class FRBInfo:
    def __init__(self, catalog1path, catalog2path=None):
        """ Load FRB catalog and data from waterfall files, both are expected to be in the same directory """
        
        csv1 = glob.glob(catalog1path+'*.csv')
        self.catalog = pd.read_csv(csv1[0])
        self.catalog['catalog'] = [1]*len(self.catalog)
        self.filepaths = glob.glob(catalog1path+'*.h5')
        
        if catalog2path:
            csv2 = glob.glob(catalog2path+'*.csv')
            catalog2 = pd.read_csv(csv2[0])
            catalog2['catalog'] = [2]*len(catalog2)
            filepaths2 = glob.glob(catalog2path+'*.h5')
            self.catalog = pd.concat([self.catalog,catalog2])
            self.filepaths += filepaths2
        
        self.catalog.drop(self.catalog[self.catalog['tns_name'] in ('FRB20190329A','FRB20200508H','FRB20200825B')].index, inplace=True)
        self.catalog['filepath'] = self.catalog['tns_name'].apply(lambda x: next((fp for fp in self.filepaths if x in fp), None))
        self.catalog.reset_index(inplace=True,drop=True)
        self.tns_names = list(set(self.catalog['tns_name'].values))
    
    def __getitem__(self, attr):
        return self.get_frb_info(attr)

    def get_frb_info(self, attr):
        """ Returns array of values for all FRBs """
        
        def read_h5(filepath, attr):
            with h5py.File(filepath, 'r') as f:
                try:
                    return f['frb'][attr][:]
                except KeyError:
                    return f['frb'].attrs[attr]
        
        if attr in self.catalog.columns:
            return list(map(N,self.catalog[attr].values))
        elif attr in ['calibrated_wfall', 'extent', 'model_spec', 'model_ts', 'model_wfall', 'plot_freq', 'plot_time', 'spec', 'ts', 'wfall', 'calibration_observation_date', 'calibration_source_name', 'dm', 'scatterfit']:
            return [read_h5(fp, attr) for fp in self.catalog['filepath'].values]
        elif attr == 'date':
            return self.catalog['mjd_inf'].apply(lambda x: Time(x, format='mjd').datetime)
        elif attr == 'n_subbursts':
            return self.catalog.groupby('tns_name')['tns_name'].transform('count')
        else:
            print(f'Attribute {attr} not found')

