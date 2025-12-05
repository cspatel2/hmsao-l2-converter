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
from sza import solar_zenith_angle

datadir = Path('/home/charmi/locsststor/proc/hmsao/l1c')
date = '20250320'
win = '6300'
fns = list(datadir.glob(f'*{date}*{win}*.nc'))
fns.sort()
# %%
ds = xr.open_mfdataset(fns)
ds = ds.assign_coords(time = (('tstamp'),[datetime.fromtimestamp(t, tz=UTC) for t in ds.tstamp.data]))
LOCATION = {'lat': 67.84080792078719, 'lon':20.410176855991722, 'elev':100 } #irf sweden
ds = ds.assign_coords( sza = ('tstamp', [solar_zenith_angle(t, lat=LOCATION['lat'],
                  lon=LOCATION['lon'], elevation=LOCATION['elev']) for t in ds.tstamp.values]))
#
# %%
fig, ax = plt.subplots(figsize=(10,6))
ds.intensity.sum('za').sum('wavelength').plot(x='time', ax = ax) #type: ignore
cax = ax.twinx()
ds.sza.plot(x='time', color='orange', ax = cax)
# ax.axvline(datetime(2025,3,20,17,30,0, tzinfo=UTC), color='red', linestyle='--')
ax.axvline(ds.time.values[-200], color='red', linestyle='--')
# %%
ds.sza.sel(tstamp=ds.tstamp.values[-200]).values
# %%
