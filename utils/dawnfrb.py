import glob
import h5py
import pickle
import numpy as np
from datetime import datetime
from cfod import catalog
catalog = catalog.as_dataframe()

# Load data from waterfall files
base_path = '/mnt/c/Users/findm/Desktop/surf2023/waterfall_data/data/'
filepaths = glob.glob(base_path+'*.h5')
tns_names = [fp.split('/')[-1].split('_')[0] for fp in filepaths]

def N(x):
    """ Convert a string to a number if possible """
    try:
        return float(x)
    except ValueError:
        return x

def get_frb_info(attr):
    """ Returns array of values for all FRBs """
    if attr == 'aligned_ts':
        return pickle.load(open('chime_frb_535_interpolated_ts.pkl','rb'))
    arr = []
    for tns_name in tns_names:
        a = None
        if attr in list(catalog.columns):
            result = catalog[catalog['tns_name'] == tns_name][attr].values
            # if len(result) > 1: print(f'Multiple sub-bursts found for {tns_name}, using value from first')
            a = N(catalog[catalog['tns_name'] == tns_name][attr].values[0].replace('<',''))
        elif attr in ['calibrated_wfall', 'extent', 'model_spec', 'model_ts', 'model_wfall', 'plot_freq', 'plot_time', 'spec', 'ts', 'wfall']:
            path = [fp for fp in filepaths if tns_name in fp][0]
            with h5py.File(path, 'r') as f:
                a = f['frb'][attr][:]
        elif attr in ['calibration_observation_date', 'calibration_source_name', 'dm', 'scatterfit', 'tns_name']:
            path = [fp for fp in filepaths if tns_name in fp][0]
            with h5py.File(path, 'r') as f:
                a = f['frb'].attrs[attr]
        elif attr == 'date':
            a = datetime.strptime(tns_name[3:-1], '%Y%m%d')
        elif attr == 'n_subbursts':
            a = len(catalog[catalog['tns_name'] == tns_name].values)
        else:
            print(f'Attribute {attr} not found for {tns_name}')
        arr.append(a)
    return arr
