#%%
import numpy as np
import pytz
import xarray as xr
from typing import Iterable
import matplotlib.pyplot as plt
from functions import get_feature_bounds
from pathlib import Path

from sza import solar_zenith_angle
from datetime import datetime, timezone
from pytz import UTC
#%%
def str2bool(value: str) -> bool:
    if value.lower() in ('true', '1', 't', 'y', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'f', 'n', 'no'):
        return False
    raise ValueError("Invalid boolean value: {}".format(value))


def rms_func(data, axis=None):
    return np.sqrt(np.sum(data**2, axis=axis))

#%%
PLOT = True
win = '5577'
datadir = Path('../data/l1c')
fns = np.sort(np.unique(list(datadir.glob(f'*{win}*.nc')))) #type: ignore

#%%
LOCATION = {'lon': 24.41, 'lat': 67.84, 'elev': 420}

#initialize
ds = xr.open_mfdataset(fns, combine='by_coords')
nds = ds.copy()
wl = int(win)/10  #wavelength in nm

# add sza TO SZA
nds['sza'] = ('tstamp', [solar_zenith_angle(t, lat=LOCATION['lat'], lon=LOCATION['lon'], elevation=LOCATION['elev']) for t in ds.tstamp.values])
nds.sza.attrs = {'units': 'deg','long_name': 'Solar Zenith Angle'}
#%%
# calc feature bounds
astro_twilight = 90 + 20  #deg
wlslice = slice(wl - 0.1, wl + 0.1)
da = nds.where(nds.sza> astro_twilight, drop = True) #nighttime only
#%%
da = da.intensity.sel(wavelength=wlslice).sum(dim='tstamp', skipna=True)
bdict = get_feature_bounds(win, da, bgoffset=0.01, prominence=0.5, rel_height=0.8, returnfullds=False)
if bool(np.isnan([v for k,v in bdict.items()]).any()):  #type: ignore
    raise ValueError(f'Could not find feature bounds for window {win} on date. Check data quality or adjust get_feature_bounds parameters.')
#%%
if PLOT:
    plt.figure()
    tidx= -170
    nds.intensity.isel(tstamp = tidx).plot(y = 'za')
    for k,v in bdict.items():
        plt.axvline(v, color='red', linestyle='--')
    plt.title(f'sza = {nds.sza.isel(tstamp = tidx).values:.2f} deg')
    plt.show()

#%%
################## l2 PROCESSING #############################

#### PREPRING DATASET ##########
# 1. remove kunnecesarry variables, add them back to final ds
sza = nds.sza
exposure = nds.exposure
ccdtemp = nds.ccdtemp
nds = nds.drop_vars(['exposure', 'ccdtemp'])

#2. bin data along za
zabinsize = 0.5  #
ZABINSIZE = int(np.ceil(zabinsize/np.mean(np.diff(nds.za.values))))
#bin 
coarsen = nds.coarsen(za=ZABINSIZE, boundary='trim')
nds = coarsen.sum( skipna=True) #type: ignore
nds = nds.assign(noise = coarsen.reduce(rms_func).noise)

# 3. separate daytime and nighttime data
civil_twilight = 90 +12  #deg
daytime_sza_cutoff = civil_twilight # sza > twilight_cutoff is night, sza <= twilight_cutoff is day 
dayds = nds.where(nds.sza <= daytime_sza_cutoff, drop=True)
nightds = nds.where(nds.sza > daytime_sza_cutoff, drop=True)

# 4. process daytime data if solar dir is provided
soldir = None
if soldir is not None:
    # get solar data file for this window and date
    solar_fnames = sorted(Path(args.soldir).glob(root_glob + f'*{date}*{win}*.nc'))
    solards = xr.open_mfdataset(solar_fnames, combine='by_coords')
    solards = solards.drop_vars(['exposure', 'ccdtemp'])
    # bin solar data along za
    solar_coarsen = solards.coarsen(za=ZABINSIZE, boundary='trim')
    solards = solar_coarsen.mean( skipna=True) #type: ignore
    solards = solards.assign(noise = solar_coarsen.reduce(rms_func).noise)

    # do solar subtraction to extract emission spectra
    ### 1.  match peaks of both spectra 
    ### 2. scale solar spectra to match peak intensity in data spectra
    ### 3. subtract scaled solar spectra from data spectra
else:
    # if no solar subtraction, set daytime data to nan
    dayds.intensity.data = np.full(dayds.intensity.shape, np.nan)
    dayds.noise.data = np.full(dayds.noise.shape, np.nan)
#%%
# 5. reconstruct the full dataset
del nds
nds = xr.concat([dayds, nightds], dim='tstamp')
nds = nds.sortby('tstamp')
# del dayds, nightds

plt.figure()
dayds.intensity.isel(tstamp=0).plot()
plt.show()
plt.figure()
nightds.intensity.isel(tstamp=0).plot()
plt.show()
#%%
#### line brightness calculation ##########

# 1. calculate line brightness for backgrounds (by sum)
bck_ = [] #list of dataarrays for backgrounds
for i in range(1,3):
    start, end = bdict[f'b{i}_start'], bdict[f'b{i}_end']
    bds = nds.sel(wavelength=slice(start, end)) #intensity is in Rayleigh/nm
    DWL = np.mean(np.diff(nds.wavelength)) #nm
    bds = bds.sum(dim='wavelength', skipna=True)
    bds['intensity'] *= DWL #intergrate over wavelength to get Rayleigh
    bds = bds.assign(noise = nds.sel(wavelength=slice(start, end)).noise.reduce(rms_func, dim='wavelength'))
    bds['noise'] *= DWL
    bck_.append(bds)
#%%
plt.figure()
for bds in bck_:
    bds.intensity.isel(tstamp = slice(None,None)).mean('za', skipna = True).plot()
# plt.axvline(bds.tstamp[-250].values)
#%%    
# average background dataarrays
bckds = xr.concat(bck_,dim = 'idx')
del bck_
bckds = bckds.assign(
    intensity = bckds.intensity.mean(dim='idx'),
    noise = bckds.noise.reduce(rms_func, dim='idx') / bckds.idx.size,
    )
#%%
bckds.intensity.isel(tstamp = 0).plot()
#%%
#2. calculate line brightness for features (by sum) and background substraction
# line_ = [] #list of dataarrays for features
start, end = bdict['start'], bdict['stop']
lds = nds.sel(wavelength=slice(start, end)) #intensity is in Rayleigh/nm
DWL = np.mean(np.diff(nds.wavelength)) #nm
lds = lds.sum(dim='wavelength', skipna=True) * DWL #intergrate over wavelength to get Rayleigh
lds = lds.assign(
    noise = nds.sel(wavelength=slice(start, end)).noise.reduce(rms_func, dim='wavelength') * DWL
    )
#noise propogation for background subtraction
noise = np.sqrt(lds.noise**2 + bckds.noise**2)
#background subtraction 
lds -= bckds.intensity

lds = lds.assign(noise = noise)
lds = lds.rename(
    {'intensity': f'{win}',
        'noise': f'{win}_err'}
)
lds[f'{win}'].attrs = {
    'units': nds.intensity.attrs['units'].replace('nm',''),
    'description': f'Line brightness of {wl} nm feature after background subtraction'
}
lds[f'{win}_err'].attrs = {
    'units': nds.intensity.attrs['units'].replace('nm',''),
    'description': f'Uncertainty in line brightness of {wl} nm feature after background subtraction'
}
#%%
bckds
#%%
##### FINAL DATASET PREPARATION AND SAVING ##########
# 1. rename variables
bckds = bckds.rename(
    {'intensity': f'bg',
        'noise': f'bg_err'}
)
bckds = bckds.assign_attrs({
    'units': 'Rayleigh',
    'long_name': 'Mean Background Brightness',
    'description': 'Background line brightness after integration over background windows'
})
bckds[f'bg_err'].attrs = {
    'units': 'Rayleigh',
    'long_name': 'Mean Background Brightness Error',
    'description': 'Error in background line brightness after integration over background windows'
}

# 2. merge line and background datasets
dslist = [lds, bckds]
saveds = xr.merge(dslist, compat='override')
#%%
if soldir is None:
    saveds[f'{win}'].sel(tstamp = dayds.tstamp.data).data = np.full((dayds.tstamp.size, dayds.za.size), np.nan)
    saveds[f'bg'].sel(tstamp = dayds.tstamp.data).data = np.full((dayds.tstamp.size, dayds.za.size), np.nan)

#%% 
#%%
saveds = saveds.assign_coords(
    dict(
        sza=('tstamp', sza.values, sza.attrs),
        ccdtemp=('tstamp', ccdtemp.values, ccdtemp.attrs),
        exposure=('tstamp', exposure.values, exposure.attrs),
    ))

print(daytime_sza_cutoff)
saveds = saveds.assign_coords(
    dict(
        daybool=xr.where(saveds.sza <= daytime_sza_cutoff, 1, 0))
        )
saveds.daybool.attrs = dict(
    unit='Bool',
    description=f'True(1) if its daytime, False(0) if its nighttime. Determined using twilight cutoff {daytime_sza_cutoff} deg.')

attrs = {k:v for k,v in ds.attrs.items() if k not in ['Description', 'note']}
saveds.attrs = attrs
saveds.attrs['Description'] = 'HMSAO data'
saveds.attrs['DataProcessingLevel'] = 'L2 - Line Brightness Data'
saveds.attrs['FileCreationDate'] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S EDT")
saveds.attrs['Location'] =  f"lat: {LOCATION['lat']} deg, lon: {LOCATION['lon']} deg, elev: {LOCATION['elev']} m"

#%%
saveds[win].plot(y = 'za',vmin = 0)
saveds['daybool'].plot()
# %%
ds = ds.assign_coords(dict(time = ('tstamp',[datetime.fromtimestamp(t, tz=pytz.timezone('Europe/Stockholm')) for t in ds.tstamp.values])))
#%%
da = ds.intensity.sum('wavelength').sum('za')
#%%

#%%
y = np.log10(da)
#%%
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot()
tax = ax.twinx()
ax.plot(ds.time.data, y)
tax.plot(ds.time.data , np.abs(nds.sza.data - 90))


# plt.axvline(da.time[-150].values)
# %%
