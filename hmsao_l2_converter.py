# %%
from ast import arg
from math import isclose
import os
import sys
from glob import glob
from dataclasses import dataclass
import hdf5plugin
from matplotlib import pyplot as plt
from matplotlib.style import available
from tqdm import tqdm
from typing import Dict, List, Tuple, Iterable
from typing import SupportsFloat as Numeric
import argparse
from pathlib import Path
import re

from time import perf_counter_ns

import numpy as np
import xarray as xr
from sza import solar_zenith_angle
from itertools import chain
from datetime import datetime
from pytz import timezone, UTC

xr.set_options(netcdf_engine_order=["h5netcdf", "netcdf4", "scipy"])

LOCALPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(LOCALPATH))
from l2_converter.l2_helper_functions import get_feature_bounds
from util_functions import apply_alignment, apply_solar_subtraction

# %%


def str2bool(value: str) -> bool:
    if value.lower() in ('true', '1', 't', 'y', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'f', 'n', 'no'):
        return False
    raise ValueError("Invalid boolean value: {}".format(value))


def rms_func(data, axis=None):
    return np.sqrt(np.nansum(data**2, axis=axis))


def find_peak_wavelength(ds: xr.DataArray, sza_cutoff: float = 108) -> float:
    try:
        # nighttime, skip daytime
        ds = ds.where(ds.sza > sza_cutoff, drop=True)
    except:
        pass
    b = ds.intensity.sum('za', skipna=True).mean('tstamp')
    return float(b.idxmax('wavelength').values)


# def get_win_from_fn(fn: str | Path) -> str:
#     if isinstance(fn, str):
#         fn = Path(fn)
#     return fn.stem.split('_')[-1].split('[')[0]


def get_win_from_fn(fn: str | Path) -> str:
    if isinstance(fn, str):
        fn = Path(fn)
    return str(re.findall(r"\d+", fn.stem)[-2])

def get_date_from_fn(fn: str | Path) -> str:
    if isinstance(fn, str):
        fn = Path(fn)
    return str(re.findall(r"\d+", fn.stem)[-3])

# def get_date_from_fn(fn: str | Path) -> str:
#     if isinstance(fn, str):
#         fn = Path(fn)
#     return fn.stem.split('_')[-2]


# %% Parser arguments

# %%
# %%

@dataclass
class L2Config:
    """config dataclass object for level 2 conversion
    Arguments:
        rootdir: Root directory of L1C data
        destdir: Root directory where L2 data will be stored.
        dest_prefix: Prefix of the saved L2 data filename.
        overwrite: If True, overwrites existing file. If false, skips processing if file exists. Defaults to False.
        zabinsize: Binsize of Zenith Angle (za | y-axis) in deg. Defaults to 1.5 deg.
        windows: Window(s) to process (list of str i.e. "1235", "3456"). Defaults to None, which processes all available windows.
        dates: Dates to process in the format YYYYMMDD  (list seperated by commas). Defaults to None, which processes all available dates.
        obs_location: Latitude (deg) Longitude (deg) Elevation (m) of the observation site.
        soldir: Directory path for Solar Spectra l1c file (.nc). If None, skips daytime data processing and sets daytime line brightness to nan.

    """    
    rootdir: str | Path
    destdir: str | Path
    dest_prefix: str
    overwrite: bool
    zabinsize: float
    windows: List[str]
    dates: List[str]
    obs_location: Tuple[float, float, float]
    soldir: str | Path | None = ''
    save: bool = True

    
def l1c_to_l2_converter(win: str, date: str, config: L2Config, root_glob: str = '') -> xr.Dataset | None:
    # LOCATION
    LOCATION = {
        'lat': float(config.obs_location[0]),
        'lon': float(config.obs_location[1]),
        'elev': float(config.obs_location[2])
    }

    yymm = datetime.strptime(date, '%Y%m%d').strftime('%Y%m')
    outfn = Path(config.destdir).joinpath(
        yymm, f"{config.dest_prefix}_{date}_{win}.nc")
    outfn.parent.mkdir(parents=True, exist_ok=True)

    # check overwrite
    if outfn.exists() and config.overwrite:
        outfn.unlink()
        print(f'{outfn} removed, overwriting...')
    elif outfn.exists() and not config.overwrite:
        print(f'{outfn} exists, skipping...')
        return None

    # get data files for this window and date
    fnames = sorted(Path(config.rootdir).glob(root_glob + f'*{date}*{win}*.nc'))

    # initialize
    ds = xr.open_mfdataset(fnames, combine='by_coords')
    nds = ds.copy()
    wl = int(win)/10  # wavelength in nm

    # add sza TO SZA
    nds['sza'] = ('tstamp', [solar_zenith_angle(t, lat=LOCATION['lat'],
                  lon=LOCATION['lon'], elevation=LOCATION['elev']) for t in ds.tstamp.values])
    nds.sza.attrs = {'units': 'deg', 'long_name': 'Solar Zenith Angle'}

    # calc feature bounds
    astro_twilight = 90 + 6  # deg
    wlslice = slice(wl - 0.1, wl + 0.1)
    da = nds.where(nds.sza > astro_twilight, drop=True)  # nighttime only
    da = da.intensity.sel(wavelength=wlslice).sum(dim='tstamp', skipna=True)
    bdict = get_feature_bounds(
        win, da, bgoffset=0.01, prominence=0.5, rel_height=0.8, returnfullds=False)
    del da
    if bool(np.isnan([v for k, v in bdict.items()]).any()):  # type: ignore
        raise ValueError(
            f'Could not find feature bounds for window {win} on date {date}. Check data quality or adjust get_feature_bounds parameters.')

    ################## l2 PROCESSING #############################

    #### PREPRING DATASET ##########
    # 1. remove kunnecesarry variables, add them back to final ds
    xlims = [val for key, val in bdict.items()]
    wlslice = slice(min(xlims), max(xlims))
    nds = nds.sel(wavelength=wlslice)
    sza = nds.sza
    exposure = nds.exposure
    ccdtemp = nds.ccdtemp
    nds = nds.drop_vars(['exposure', 'ccdtemp'])

    # 2. bin data along za
    ZABINSIZE = int(np.ceil(config.zabinsize/np.mean(np.diff(nds.za.values))))
    # bin
    coarsen = nds.coarsen(za=ZABINSIZE, boundary='trim')
    nds = coarsen.sum(skipna=True)  # type: ignore
    nds = nds.assign(noise=coarsen.reduce(rms_func).noise)

    # 3. separate daytime and nighttime data
    civil_twilight = 90 + 6  #deg
    # sza > twilight_cutoff is night, sza <= twilight_cutoff is day
    daytime_sza_cutoff = civil_twilight
    dayds = nds.where(nds.sza <= daytime_sza_cutoff, drop=True)
    nightds = nds.where(nds.sza > daytime_sza_cutoff, drop=True)

    # 4. process daytime data if solar dir is provided
    if config.soldir is not None:
        # get solar data file for this window and date
        solar_fnames = sorted(Path(config.soldir).glob(f'*{win}*.nc'))
        solards = xr.open_mfdataset(solar_fnames, combine='by_coords')
        solards = solards.sel(wavelength=wlslice)
        solards = solards.drop_vars(['exposure', 'ccdtemp', 'za'])
        # bin solar data along za
        solar_coarsen = solards.coarsen(za=ZABINSIZE, boundary='trim')
        solards = solar_coarsen.mean(skipna=True) #type: ignore
        solards = solards.assign(noise = solar_coarsen.reduce(rms_func).noise)
        solards = solards.mean('tstamp', skipna=True)
        intensity_attrs = solards.intensity.attrs
        noise_attrs = solards.noise.attrs

        # do solar subtraction to extract emission spectra
        # 1.  match peaks of both spectra
        a = apply_alignment(solards, dayds.copy(), -0.015)
        solards['intensity'] = a.mean('tstamp', skipna=True)
        # 2. scale solar spectra to match peak intensity in data spectra
        # 3. subtract scaled solar spectra from data spectra
        
        s = apply_solar_subtraction(solards, dayds)
        dayds['intensity'] = s
        dayds.intensity.attrs  = intensity_attrs
        dayds.noise.attrs  = noise_attrs
    else:
        print('No solar directory provided. Skipping daytime data processing.')
        # # if no solar subtraction, set daytime data to nan
        dayds.intensity.data = np.full(dayds.intensity.shape, np.nan)
        dayds.noise.data = np.full(dayds.noise.shape, np.nan)

    # 5. reconstruct the full dataset
    del nds
    nds = xr.concat([dayds, nightds], dim='tstamp')
    nds = nds.sortby('tstamp')
    nds.intensity.attrs = ds.intensity.attrs
    nds.noise.attrs = ds.noise.attrs
    # del dayds, nightds

    #### line brightness calculation ##########

    # 1. calculate line brightness for backgrounds (by sum)
    bck_ = []  # list of dataarrays for backgrounds
    for i in range(1, 3):
        start, end = bdict[f'b{i}_start'], bdict[f'b{i}_end']
        # intensity is in Rayleigh/nm
        bds = nds.sel(wavelength=slice(start, end))
        DWL = np.mean(np.diff(nds.wavelength))  # nm
        # intergrate over wavelength to get Rayleigh
        bds = bds.sum(dim='wavelength', skipna=True) * DWL
        bds = bds.assign(noise=nds.sel(wavelength=slice(
            start, end)).noise.reduce(rms_func, dim='wavelength') * DWL)
        bck_.append(bds)
    # average background dataarrays
    bckds = xr.concat(bck_, dim='idx')
    del bck_
    bckds = bckds.assign(
        intensity=bckds.intensity.mean(dim='idx'),
        noise=bckds.noise.reduce(rms_func, dim='idx') / bckds.idx.size,
    )

    # 2. calculate line brightness for features (by sum) and background substraction
    # line_ = [] #list of dataarrays for features
    start, end = bdict['start'], bdict['stop']
    lds = nds.sel(wavelength=slice(start, end))  # intensity is in Rayleigh/nm
    DWL = np.mean(np.diff(nds.wavelength))  # nm
    # intergrate over wavelength to get Rayleigh
    lds = lds.sum(dim='wavelength', skipna=True) * DWL
    lds = lds.assign(
        noise=nds.sel(wavelength=slice(start, end)).noise.reduce(
            rms_func, dim='wavelength') * DWL
    )
    # noise propogation for background subtraction
    noise = np.sqrt(lds.noise**2 + bckds.noise**2)
    # background subtraction
    lds -= bckds.intensity

    lds = lds.assign(noise=noise)
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
    # line_.append(lds)

    ##### FINAL DATASET PREPARATION AND SAVING ##########
    # 1. rename variables
    bckds = bckds.rename(
        {'intensity': f'bg',
            'noise': f'bg_err'}
    )
    bckds = bckds.assign_attrs({
        'units': nds.intensity.attrs['units'].replace('nm',''),
        'long_name': 'Mean Background Brightness',
        'description': 'Background line brightness after integration over background windows'
    })
    bckds[f'bg_err'].attrs = {
        'units': nds.intensity.attrs['units'].replace('nm',''),
        'long_name': 'Mean Background Brightness Error',
        'description': 'Error in background line brightness after integration over background windows'
    }

    # 2. merge line and background datasets
    dslist = [lds, bckds]
    saveds = xr.merge(dslist, compat='override')

    if config.soldir is None: # make sure the daytimne data is nan if no solar data provided
        saveds[f'{win}'].sel(tstamp = dayds.tstamp.data).data = np.full((dayds.tstamp.size, dayds.za.size), np.nan)
        saveds[f'bg'].sel(tstamp = dayds.tstamp.data).data = np.full((dayds.tstamp.size, dayds.za.size), np.nan)

    saveds = saveds.assign_coords(
        dict(
            sza=('tstamp', sza.values, sza.attrs),
            ccdtemp=('tstamp', ccdtemp.values, ccdtemp.attrs),
            exposure=('tstamp', exposure.values, exposure.attrs),
        ))

    saveds = saveds.assign_coords(
        dict(
            daybool=xr.where(saveds.sza < daytime_sza_cutoff, 1, 0))
    )
    saveds.daybool.attrs = dict(
        unit='Bool',
        description=f'True(1) if its daytime, False(0) if its nighttime. Determined using twilight cutoff {daytime_sza_cutoff} deg.')

    attrs = {k: v for k, v in ds.attrs.items() if k not in [
        'Description', 'note']}
    saveds.attrs = attrs
    saveds.attrs['Description'] = 'HMSAO data'
    saveds.attrs['DataProcessingLevel'] = 'L2 - Line Brightness Data'
    saveds.attrs['FileCreationDate'] = datetime.now().strftime(
        "%m/%d/%Y, %H:%M:%S EDT")
    saveds.attrs['Location'] = f"lat: {LOCATION['lat']} deg, lon: {LOCATION['lon']} deg, elev: {LOCATION['elev']} m"

    if config.save:
        # 3. save dataset
        encoding = {var: hdf5plugin.Zstd(clevel=3)
                    for var in (*saveds.data_vars.keys(), *saveds.coords.keys())}
        print('Saving %s...\t' % (os.path.basename(outfn)), end='')
        sys.stdout.flush()
        tstart = perf_counter_ns()
        saveds.to_netcdf(outfn, encoding=encoding)
        tend = perf_counter_ns()
        print(f'Done. [{(tend-tstart)*1e-9:.3f} s]')
        del saveds, nds, ds
    else:
        return saveds
# %%

def main(config: L2Config):
    """ converts L1C files to L2 files. This includes:
     1. calculating line intensities of each spectral feature.
     2. background subtraction using the defined background windows and calculating background line intensity.
     3. adding new attributes like solar zenith angle, exposure time, ccd temperature, and a boolean for day/night.

    Args:
        config (L2Config): Configuration dataclass for L2 conversion.
            arguments:
                - rootdir: Root directory of L1C data
                - destdir: Root directory where L2 data will be stored.
                - dest_prefix: Prefix of the saved L2 data filename.
                - overwrite: If True, overwrites existing file. If false, skips processing if file exists. Defaults to False.
                - zabinsize: Binsize of Zenith Angle (za | y-axis) in deg. Defaults to 1.5 deg.
                - windows: Window(s) to process (list of str i.e. "1235", "3456"). Defaults to None, which processes all available windows.
                - dates: Dates to process in the format YYYYMMDD  (list seperated by commas). Defaults to None, which processes all available dates.
                - obs_location: Latitude (deg) Longitude (deg) Elevation (m) of the observation site.
                - soldir: Directory path for Solar Spectra l1c file (.nc). If None, skips daytime data processing and sets daytime line brightness to nan.

    Raises:
        ValueError: Invalid boolean value for overwrite argument.
        ValueError: Root directory does not exist or is not a directory.
        ValueError: No L1C files found in root directory or subdirectories.
        ValueError: Solar directory does not exist or is not a directory.
        ValueError: No Solar L1C files found in solar directory or subdirectories.
        ValueError: Could not find feature bounds for a given window and date. Check data quality or adjust get_feature_bounds parameters.  
    """    

    ############## CHECK DIRECTORIES ##############
    # DESTINATION DIR
    if isinstance(config.destdir, str):
        config.destdir = Path(config.destdir)
    config.destdir = config.destdir.expanduser()
    config.destdir.mkdir(parents=True, exist_ok=True)

    # ROOT DIR
    if isinstance(config.rootdir, str):
        config.rootdir = Path(config.rootdir)
    config.rootdir = config.rootdir.expanduser()
    if not config.rootdir.exists():
        raise ValueError(f'Root directory {config.rootdir} does not exist.')
    elif not config.rootdir.is_dir():
        raise ValueError(f'Root path {config.rootdir} is not a directory.')
    # check for data
    root_glob = ''
    l1c_files = sorted(config.rootdir.glob('*.nc'))
    if len(l1c_files) < 1:  # no.nc files in rootdir, check if subdirs havee .nc files
        root_glob = '**/'
        l1c_files = sorted(config.rootdir.glob(root_glob + '*.nc'))
        if len(l1c_files) < 1:  # no .nc files in subdirs either
            raise ValueError(
                f'No L1C files found in root directory {config.rootdir} or subdirectories.')
        else:
            ...  # if subdirs needed, make the list here

    # SOLAR DIR
    if config.soldir == '':
        print('No solar directory provided. No daytime data will be processed.')
        config.soldir = None
    elif isinstance(config.soldir, str):
        config.soldir = Path(config.soldir)
        config.soldir = config.soldir.expanduser()
        if not config.soldir.exists():  # does it exists?
            raise ValueError(f'Solar directory {config.soldir} does not exist.')
        elif not config.soldir.is_dir():  # is it a dir?
            raise ValueError(f'Solar path {config.soldir} is not a directory.')
        else:  # exits and is dir, check for .nc files
            solar_glob = ''
            solar_files = sorted(config.soldir.glob(solar_glob+'*.nc'))
            if len(solar_files) < 1:  # no .nc files in rootdir,  check if subdirs havee .nc files
                solar_glob = '**/'
                solar_files = sorted(config.soldir.glob(solar_glob+'*.nc'))
                if len(solar_files) < 1:  # no .nc files in subdirs either
                    raise ValueError(
                        f'No Solar L1C files found in solar directory {config.soldir} or subdirectories.')
                else:
                    ...  # if subdirs needed, make the list here

    ############## PROCESSING PARAMETERS ##############
    # WINDOWS TO PROCESS
    available_windows = np.unique([get_win_from_fn(f) for f in l1c_files])

    if config.windows == [''] or config.windows is None:
        print('No windows provided. Processing all available windows.')
        valid_windows = available_windows.tolist()
    else:
        valid_windows =list(set(config.windows) & set(available_windows))
    config.windows = valid_windows # reassign to config for later use

    print(f'Processing windows: {config.windows}')

    # DATES TO PROCESS
    all_valid_dates = np.unique([get_date_from_fn(f) for f in l1c_files])
    print(f'Available dates in data: {all_valid_dates}')
    if config.dates == [''] or config.dates is None:
        print('No dates provided. Processing all available dates.')
        valid_dates = all_valid_dates
    else:
        valid_dates = list(set(config.dates) & set(all_valid_dates))
    config.dates = valid_dates  # reassign to config for later use
    print(f'Processing dates: {config.dates}')

    # DESTINATION PREFIX
    # if no prefix provided, use rootdir name
    if config.dest_prefix is None or config.dest_prefix == '':
        dest_prefix = 'hmsao-l2'
        print(
            f'No destination prefix provided. Using default prefix: {dest_prefix}')
    else:
        dest_prefix = config.dest_prefix
        if 'l2' not in config.dest_prefix.lower():
            dest_prefix += '_l2'
    config.dest_prefix = dest_prefix  # reassign to config for later use

    # ZENITH ANGLE BINSIZE
    if config.zabinsize is None:
        config.zabinsize = 1.5
    else:
        config.zabinsize = float(config.zabinsize)

    for win in tqdm(config.windows):
        for date in config.dates:
            l1c_to_l2_converter(win, date, config, root_glob=root_glob)



# %%

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description='Convert L1C data (calibrated images) to L2 data (line brightness).')
    parser.add_argument('rootdir',
        metavar='rootdir',
        type=str,
        help='Root directory of L1A data'
    )
    parser.add_argument('destdir',
        metavar='destdir',
        # required = False,
        type=str,
        default=Path.cwd(),
        nargs='?',
        help='Root directory where L1 data will be stored.'
    )
    parser.add_argument('dest_prefix',
        metavar='dest_prefix',
        # required = False,
        type=str,
        default=None,
        nargs='?',
        help='Prefix of the saved L1 data finename.'
    )
    parser.add_argument('--overwrite',
        required=False,
        type=str2bool,
        default=False,
        nargs='?',
        help='If True, overwrites existing file. If false, skips processing if file exists. Defaults to False.'
    )
    parser.add_argument('--zabinsize',
        type=float,
        default=None,
        nargs='?',
        help='bBinsize of Zenith Angle (za | y-axis) in deg. Defaults to 1.5 deg.'
    )
    def list_of_strings(arg: str) -> List[str]:
        return arg.split(',')
    parser.add_argument('--windows',
        # metavar = 'NAME',
        # action='append',
        required=False,
        type=list_of_strings,
        default=None,
        nargs='?',
        help='Window(s) to process (list of str i.e. "1235", "3456").'
    )
    parser.add_argument('--dates',
        required=False,
        type=list_of_strings,
        default=None,
        nargs='?',
        help='Dates to process in the format YYYYMMDD  (list seperated by commas).'
    )
    parser.add_argument('--location',
        required=True,
        type=float,
        nargs=3,
        help='Latitude (deg) Longitude (deg) Elevation (m) of the observation site.'
    )
    parser.add_argument('--soldir',
        required=False,
        type=str,
        default=None,
        nargs='?',
        help='Directory path for Solar Spectra l1c file (.nc).'
    )
    
    args = parser.parse_args()

    config = L2Config(
        rootdir=args.rootdir,
        destdir=args.destdir,
        dest_prefix=args.dest_prefix,
        overwrite=args.overwrite,
        zabinsize=args.zabinsize,
        windows=args.windows,
        dates=args.dates,
        obs_location=tuple(args.location),
        soldir=args.soldir
    )
    main(config)

