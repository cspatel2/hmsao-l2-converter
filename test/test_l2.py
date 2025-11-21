#%%
import numpy as np
import xarray as xr
from typing import Iterable
import matplotlib.pyplot as plt
from functions import get_feature_bounds
from pathlib import Path
from sza import solar_zenith_angle
#%%
def str2bool(value: str) -> bool:
    if value.lower() in ('true', '1', 't', 'y', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'f', 'n', 'no'):
        return False
    raise ValueError("Invalid boolean value: {}".format(value))


def rms_func(data, axis=None):
    return np.sqrt(np.sum(data**2, axis=axis))


win = '5577'
datadir = Path('../data/l1c')
fns = np.sort(np.unique(list(datadir.glob(f'*{win}*.nc')))) #type: ignore
#%%
ds = xr.open_dataset(fns[0])
da = ds['intensity'].sum(dim = 'tstamp')
bdict = get_feature_bounds(win, da, bgoffset=0.01, prominence=0.5, rel_height=0.8, returnfullds=False)
# da.plot()
# plt.axvline(bdict['start'], color='red')
# plt.axvline(bdict['stop'], color='red')
# plt.axvline(bdict['b1_start'], color='green')
# plt.axvline(bdict['b1_end'], color='green')
# plt.axvline(bdict['b2_start'], color='blue')
# plt.axvline(bdict['b2_end'], color='blue')
del da

