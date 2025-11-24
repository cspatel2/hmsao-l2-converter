#%%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from pytz import UTC
# %%
datadir = Path('../data/l2')
fig, ax = plt.subplots(2,1,figsize=(10,6), sharex=True)

for i, val in enumerate([('5577','Greens'), ('6300','Reds')]):
    win, color = val
    fns = np.sort(np.unique(list(datadir.glob(f'*/*{win}*.nc')))) #type: ignore
    ds = xr.open_dataset(fns[0])
    ds = ds.assign_coords(time = (('tstamp'),[datetime.fromtimestamp(t, tz=UTC) for t in ds.tstamp.data]))
    da = ds[win]
    # da.data = np.log10(da.data)
    # da.plot(y = 'za', x = 'time',vmin = 6 ,cmap=color, ax = ax[i]) #type: ignore

    vmin  = np.nanpercentile(da.data, 10)
    vmax  = np.nanpercentile(da.data, 99)
    da.plot(y = 'za', x = 'time', vmin = vmin, vmax = vmax, cmap=color, ax = ax[i]) #type: ignore
ax[0].set_xlabel('')
# %%
