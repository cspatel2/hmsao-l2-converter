#%%
from cProfile import label
from turtle import left
from unittest import skip
from matplotlib import colorbar
import numpy as np
import pytz
import xarray as xr
from typing import Iterable
import matplotlib.pyplot as plt
from l2_converter.l2_helper_functions import get_feature_bounds
from pathlib import Path

from sza import solar_zenith_angle
from datetime import datetime, timezone
from pytz import UTC
from util_functions import apply_alignment, apply_solar_subtraction
#%%
def str2bool(value: str) -> bool:
    if value.lower() in ('true', '1', 't', 'y', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'f', 'n', 'no'):
        return False
    raise ValueError("Invalid boolean value: {}".format(value))


def rms_func(data, axis=None):
    return np.sqrt(np.nansum(data**2, axis=axis))

#%%
PLOT = True
win = '6300'
datadir = Path('../data/l1c')
fns = np.sort(np.unique(list(datadir.glob(f'*{win}*.nc')))) #type: ignore

#%%
LOCATION = {'lon': 24.41, 'lat': 67.84, 'elev': 420}

#initialize
ds = xr.open_mfdataset(fns, combine='by_coords')
nds = ds.copy()
wl = int(win)/10  #wavelength in nm
#%%
PLOT = True
if PLOT:
    wlslice = slice(wl - .6, wl + 1)
    da = nds.intensity.sel(wavelength = wlslice).isel(tstamp = 400)
    plt.figure(figsize=(4,3))
    da.plot(y='za', cbar_kwargs={'label':'[Photons/s.cm2.sr.nm]'})
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Zenith Angle [deg]')
    dt = datetime.fromtimestamp(float(da.tstamp.values), timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    plt.title(f'{dt} UTC')
    plt.show()

#%%
# add sza TO SZA
nds['sza'] = ('tstamp', [solar_zenith_angle(t, lat=LOCATION['lat'], lon=LOCATION['lon'], elevation=LOCATION['elev']) for t in ds.tstamp.values])
nds.sza.attrs = {'units': 'deg','long_name': 'Solar Zenith Angle'}
#%%
# calc feature bounds
astro_twilight = 90 + 20  #deg
wlslice = slice(wl - 0.1, wl + 0.1)
single = nds.where(nds.sza> astro_twilight, drop = True) #nighttime only
#%%
single = single.intensity.sel(wavelength=wlslice).sum(dim='tstamp', skipna=True)
bdict = get_feature_bounds(win, single, bgoffset=0.01, prominence=0.5, rel_height=0.8, returnfullds=False)
if bool(np.isnan([v for k,v in bdict.items()]).any()):  #type: ignore
    raise ValueError(f'Could not find feature bounds for window {win} on date. Check data quality or adjust get_feature_bounds parameters.')
#%%
PLOT = False
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
# 3. separate daytime and nighttime data
civil_twilight = 90 + 6  #deg
# sza > twilight_cutoff is night, sza <= twilight_cutoff is day
daytime_sza_cutoff = civil_twilight
dayds = nds.where(nds.sza <= daytime_sza_cutoff, drop=True)
nightds = nds.where(nds.sza > daytime_sza_cutoff, drop=True)

#%% # 4. process daytime data if solar dir is provided
soldir = '/home/charmi/locsststor/proc/hmsao_solspec/l1c'
if soldir is not None:
    solar_fnames = sorted(Path(soldir).glob(f'*{win}*.nc'))
    solards = xr.open_mfdataset(solar_fnames, combine='by_coords')
    solards = solards.drop_vars(['exposure', 'ccdtemp', 'za'])
    # bin solar data along za
    solar_coarsen = solards.coarsen(za=ZABINSIZE, boundary='trim')
    solards = solar_coarsen.mean(skipna=True) #type: ignore
    solards = solards.assign(noise = solar_coarsen.reduce(rms_func).noise)
    solards = solards.mean('tstamp', skipna=True)
    intensity_attrs = solards.intensity.attrs
    noise_attrs = solards.noise.attrs
b = apply_alignment(solards, dayds.copy(), -0.015)
#%%

solards['intensity'] = b.mean('tstamp', skipna=True)
#%%
#%%
xlims = [val for key, val in bdict.items()]
wlslice = slice(min(xlims), max(xlims))
sol = solards.sel(wavelength = wlslice)
sky = dayds.intensity.sel(wavelength = wlslice)
#%%
sub = apply_solar_subtraction(sol, sky)
# %%
sub.isel(tstamp = 200).plot()
#%%
dayds['intensity'] = sub
#%%
# if PLOT:
#     tidx = 50
#     nsol = solards.intensity.isel(tstamp =tidx, za = 10)
#     nsol -= nsol.min()
#     nsol /= nsol.max()
#     nsky = dayds.intensity.isel(tstamp =tidx, za = 10)
#     nsky -= nsky.min()
#     nsky /= nsky.max()
#     plt.figure()
#     wslice = slice(629.5, 630.5)
#     nsol.sel(wavelength = wslice).plot(label='solar', color = 'orange')
#     nsky.sel(wavelength = wslice).plot(label='sky', color = 'blue')
#     plt.legend()

# # %%
# sol = solards.mean('tstamp', skipna=True)
# sky = dayds
# #%%
# PLOT = True
# if PLOT:
#     a = apply_alignment(sol, sky, -0.015)
#     tidx = 40
#     # nsol = a.isel(tstamp =tidx, za = 10)
#     nsol = a.mean('tstamp').isel(za = 10)
#     nsol -= nsol.min()
#     nsol /= (nsol.max())
#     nsky = dayds.intensity.isel(tstamp =tidx, za = 10)
#     nsky -= nsky.min()
#     nsky /= ( nsky.max())
#     plt.figure()
#     wslice = slice(629.5, 630.5)
#     nsol.sel(wavelength = wslice).plot(label='solar', color = 'orange')
#     nsky.sel(wavelength = wslice).plot(label='sky', color = 'blue')
#     plt.legend()
# #%%
# xlim = [val for k,val in bdict.items()]
# wlslice = slice(np.min(xlim), np.max(xlim))
# sol = a.mean('tstamp').sel(wavelength = wlslice)
# sky = dayds.intensity.sel(wavelength = wlslice)

#     #%%
#     def calc_sclaing_factor(sky, sol):
#         denom = np.dot(sky, sky)
#         if denom == 0:
#             return 1
#         else:
#             return np.dot(sky, sol) / denom
    
#     def calc_scaling_factor_arr(sky1,sol1,wls1,sky2,sol2,wls2):
#         sf1 = sky1/sol1
#         sf2 = sky2/sol2
#         return list(sf1)+list(sf2), list(wls1) + list(wls2)


#     #%%
#     xlim = [val for k,val in bdict.items()]
#     wlslice = slice(np.min(xlim), np.max(xlim))
#     tsky =  dayds.intensity.isel(tstamp =tidx, za = 10).sel(wavelength = wlslice)
#     tsol = a.isel(tstamp =tidx, za = 10). sel(wavelength = wlslice)
#     mask = np.isfinite(tsky.values) & np.isfinite(tsol.values)
#     sky_masked = tsky[mask]
#     sol_masked = tsol[mask]

#     #%%
#     n = 3
#     leftmask =np.arange(0,30,n)
#     rightmask = np.append(np.arange(-30,-1,n),-1)
#     rsf, rwl = calc_scaling_factor_arr(
#         sky_masked[leftmask].values,
#         sol_masked[leftmask].values,
#         sky_masked.wavelength.values[leftmask],
#         sky_masked[rightmask].values,
#         sol_masked[rightmask].values,
#         sky_masked.wavelength.values[rightmask],)
#     sf = np.interp(sky_masked.wavelength.values, rwl, rsf)
# #%%
#     fig,(ax, ax1) = plt.subplots(2,1, figsize=(7,4.5),sharex=True, layout = 'constrained')
#     l1, = sol_masked.plot(ax=ax, label='Solar', color = 'orange')
#     scaled_sky = sky_masked.copy() *1/ sf
#     l2, = scaled_sky.plot(ax=ax, color = 'green', linestyle='--', label='Scaled Sky (interp)')
#     ax.set_ylim(None, sol_masked.max()*1.1)
#     ax.legend(loc = 'upper left')
#     ax.set_ylabel('[Photons/s.cm².sr.nm]')
#     ax.axvline(bdict['start'], color='black', linestyle='--', lw = 0.4)
#     ax.axvline(bdict['stop'], color='black', linestyle='--', lw = 0.4)
#     cax = ax.twinx()
#     l3, = sky_masked.plot(ax=cax, label='Sky', color = 'blue')
#     cax.set_ylim(None, 3e11)
#     cax.legend()
#     ax.set_title('')
#     cax.tick_params(axis='y', labelcolor='blue')
#     cax.set_ylabel('[Photons/s.cm².sr.nm]', color= 'blue')
#     dt = datetime.fromtimestamp(float(sky_masked.tstamp.values), timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
#     cax.set_title(dt + f' | sza = {dayds.sza.isel(tstamp = tidx).values:.2f} deg')

#     res = scaled_sky - sol_masked
#     std = res.std().values
#     l1, = res.plot(ax=ax1, label='Atms Emission = Scaled Sky - Solar', color = 'Red')
#     f1 = ax1.fill_between(res.wavelength.values, res.values - std/2, res.values + std/2, color='red', alpha=0.2)
#     ax1.axvline(bdict['start'], color='black', linestyle='--', lw = 0.4)
#     ax1.axvline(bdict['stop'], color='black', linestyle='--', lw = 0.4)
#     ax1.set_ylim(0, res.max()*1.1)
#     ax1.set_ylabel('[Photons/s.cm².sr.nm]')

#     ax1.legend(handles=[(l1, f1)], labels=['Atms Emission ± 1σ '])
#     ax1.set_title('')
#     plt.savefig('solar_subtraction_example.png', dpi=300)

#%%

#     norm_night = nightds.intensity.isel(tstamp = tidx, za = 10).sel(wavelength = wslice)
#     norm_res = residual
#     norm_night -= norm_night.min()
#     norm_night /= norm_night.max()
#     norm_res -= norm_res.min()
#     norm_res /= norm_res.max()
#     norm_night.plot(label='night', color = 'black', linestyle='--', lw = 0.5)
#     # residual.plot(label='residual', color = 'green')
#     norm_res.plot(label='residual', color = 'green')
#     for k,v in bdict.items():
#         plt.axvline(v, color='red', linestyle='--', lw = 0.4)
#         plt.title(f'sza = {nds.sza.isel(tstamp = tidx).values:.2f} deg')

#     #%%
#     residual.plot(label='residual', color = 'green')
#     plt.ylim(0, residual.max()*1.1)
#     plt.xlim(629.98, 630.12)

# sub = apply_solar_subtraction(sol, sky)
#%%
#     # do solar subtraction to extract emission spectra
#     ### 1.  match peaks of both spectra 
#     ### 2. scale solar spectra to match peak intensity in data spectra
#     ### 3. subtract scaled solar spectra from data spectra
# else: 
#     # if no solar subtraction, set daytime data to nan
#     dayds.intensity.data = np.full(dayds.intensity.shape, np.nan)
#     dayds.noise.data = np.full(dayds.noise.shape, np.nan)
#%%
# 5. reconstruct the full dataset
del nds
nds = xr.concat([dayds, nightds], dim='tstamp')
nds = nds.sortby('tstamp')
# del dayds, nightds
#%%
plt.figure(figsize=(4,3))
wlslice = slice(wl - .6, wl + 1)
dayds.intensity.sel(wavelength=wlslice).isel(tstamp=slice(50,70)).sum(dim='tstamp').plot(cbar_kwargs={'label':'Photons/s.cm2.sr.nm'})
dt = datetime.fromtimestamp(float(dayds.tstamp[200].values), timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
plt.title(dt)
plt.xlabel('Wavelength [nm]')
plt.ylabel('Zenith Angle [deg]')
plt.show()
#%%
plt.figure(figsize=(4,3))
nightds.intensity.sel(wavelength=wlslice).isel(tstamp=10).plot(cbar_kwargs={'label':'Photons/s.cm2.sr.nm'})
dt = datetime.fromtimestamp(float(nightds.tstamp[40].values), timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
plt.title(dt)
plt.xlabel('Wavelength [nm]')
plt.ylabel('Zenith Angle [deg]')
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
    'units': ds.intensity.attrs['units'].replace('nm',''),
    'description': f'Line brightness of {wl} nm feature after background subtraction'
}
lds[f'{win}_err'].attrs = {
    'units': ds.intensity.attrs['units'].replace('nm',''),
    'description': f'Uncertainty in line brightness of {wl} nm feature after background subtraction'
}

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
import matplotlib.dates as mdates
import matplotlib.colors as mcolors

fontsize = 12
plt.rcParams.update({'font.size': fontsize,
                        'axes.labelsize': fontsize,
                        'axes.titlesize': fontsize, 
                        'legend.fontsize': fontsize,
                        'figure.titlesize': fontsize})

import matplotlib.units as munits
converter = mdates.ConciseDateConverter()
munits.registry[np.datetime64] = converter
munits.registry[datetime.date] = converter
munits.registry[datetime] = converter

# single[win] = np.log10(saveds[win])
#%#
arr = saveds[win].values
#%%
plotarr = xr.DataArray(
    data = arr,
    coords = saveds[win].coords ,
    dims = saveds[win].dims,
    attrs = saveds[win].attrs,
)

plotarr = plotarr.assign_coords(
    {
        'time': (('tstamp'), [datetime.fromtimestamp(float(t), UTC) for t in plotarr.tstamp.values])
    }
)
norm = mcolors.LogNorm(vmin=1e2, vmax=1e12)
#%%
# single[win].plot(x='time',y='za', vmin = 0, cmap='Reds')
fig,ax = plt.subplots(figsize=(12,6))
plotarr.plot(x = 'time', y = 'za', cmap='Reds', norm=norm, ax = ax)
# saveds['daybool'].plot(ax = ax, x='tstamp', y='za', add_colorbar=False, cmap='Greys', alpha=0.3)
# %%

#%%
single = ds.intensity.sum('wavelength').sum('za')
#%%

#%%
y = np.log10(single)
#%%
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot()
tax = ax.twinx()
ax.plot(ds.time.data, y)
tax.plot(ds.time.data , np.abs(nds.sza.data - 90))


# plt.axvline(da.time[-150].values)
# %%
