#%% 
from math import isclose
import sys
from glob import glob
from tqdm import tqdm
from typing import Dict, List, Tuple, Iterable
from typing import SupportsFloat as Numeric
import argparse
from pathlib import Path
from functions import get_feature_bounds
from time import perf_counter_ns

import numpy as np
import xarray as xr
from sza import solar_zenith_angle
from itertools import chain
from datetime import datetime
from pytz import timezone, UTC

#%%
def str2bool(value: str) -> bool:
    if value.lower() in ('true', '1', 't', 'y', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'f', 'n', 'no'):
        return False
    raise ValueError("Invalid boolean value: {}".format(value))


def rms_func(data, axis=None):
    return np.sqrt(np.sum(data**2, axis=axis))

def find_peak_wavelength(ds:xr.DataArray,sza_cutoff:float=108) -> float:
    try:
        ds = ds.where(ds.sza > sza_cutoff, drop=True)  # nighttime, skip daytime
    except:
        pass
    b = ds.intensity.sum('za').mean('tstamp')
    return float(b.idxmax('wavelength').values)

def get_win_from_fn(fn:str | Path) -> str:
    if isinstance(fn, str):
        fn = Path(fn)
    return fn.stem.split('_')[-1].split('[')[0]

def get_date_from_fn(fn:str | Path) -> str:
    if isinstance(fn, str):
        fn = Path(fn)
    return fns[0].stem.split('_')[-2]


#%% Parser arguments
parser = argparse.ArgumentParser(
    description='Convert L1C data (calibrated images) to L2 data (line brightness).')

parser.add_argument(
    'rootdir',
    metavar='rootdir',
    type=str,
    help='Root directory of L1A data'
)

parser.add_argument(
    'destdir',
    metavar='destdir',
    # required = False,
    type=str,
    default=Path.cwd(),
    nargs='?',
    help='Root directory where L1 data will be stored.'
)

parser.add_argument(
    'dest_prefix',
    metavar='dest_prefix',
    # required = False,
    type=str,
    default=None,
    nargs='?',
    help='Prefix of the saved L1 data finename.'
)

parser.add_argument(
    '--overwrite',
    required=False,
    type=str2bool,
    default=False,
    nargs='?',
    help='If True, overwrites existing file. If false, skips processing if file exists. Defaults to False.'
)

parser.add_argument(
    '--zabinsize',
    type = float,
    default = None,
    nargs = '?',
    help = 'bBinsize of Zenith Angle (za | y-axis) in deg. Defaults to 1.5 deg.'
)

def list_of_strings(arg: str) -> List[str]:
    return arg.split(',')

parser.add_argument(
    '--windows',
    # metavar = 'NAME',
    # action='append',
    required=False,
    type=list_of_strings,
    default=None,
    nargs='?',
    help='Window(s) to process (list of str i.e. "1235", "3456").'
)

parser.add_argument(
   '--dates',
    required= False,
    type=list_of_strings,
    default = None,
    nargs = '?',
    help = 'Dates to process in the format YYYYMMDD  (list seperated by commas).'
)

parser.add_argument(
    '--location',
    required=True,
    type= float,
    nargs=3,
    help='Latitude (deg) Longitude (deg) Elevation (m) of the observation site.'
)

parser.add_argument(
    '--soldir',
    required=False,
    type=str,
    default=None,
    nargs='?',
    help='Directory path for Solar Spectra l1c file (.nc).'
)
#%%
def l1c_to_l2_converter(win, args, root_glob:str=''):
    # LOCATION
    LOCATION = {
        'lat': float(args.location[0]),
        'lon': float(args.location[1]),
        'elev': float(args.location[2])
    }

    for date in args.dates:
        yymm = datetime.strptime(date, '%Y%m%d').strftime('%y%m')
        outfn = Path(args.destdir).joinpath(yymm, f"{args.dest_prefix}_{date}_{win}.nc")
        outfn.parent.mkdir(parents=True, exist_ok=True)

        #check overwrite
        if outfn.exists() and not args.overwrite:
            outfn.unlink()
            print(f'{outfn} removed, overwriting...')
        
        #get data files for this window and date
        fnames = sorted(Path(args.rootdir).glob(root_glob + f'*{date}*{win}*.nc'))

        #initialize
        ds = xr.open_mfdataset(fnames, combine='by_coords')
        nds = ds.copy()
        wl = int(win)/10  #wavelength in nm

        # add sza TO SZA
        nds['sza'] = ('tstamp', [solar_zenith_angle(t, lat=LOCATION['lat'], lon=LOCATION['lon'], elevation=LOCATION['elev']) for t in ds.tstamp.values])
        nds.sza.attrs = {'units': 'deg','long_name': 'Solar Zenith Angle'}

        # calc feature bounds
        astro_twlight = 108  #deg
        da = nds['intensity'].sel(sza=slice(astro_twlight, None)).sum(dim = 'tstamp') #nighttime only
        bdict = get_feature_bounds(win, da, bgoffset=0.01, prominence=0.5, rel_height=0.8, returnfullds=False)
        del da

        ########## START HERE WITH SEPERATING DAY AND NIGHT HERE #########
         # not sure which version of the 1lb converter its in





    bdict = get_feature_bounds(win, da, bgoffset=0.01, prominence=0.5, rel_height=0.8, returnfullds=False)

    ...

#%%
def main():
    args = parser.parse_args()

    ############## CHECK DIRECTORIES ##############
    
    # DESTINATION DIR
    destdir = Path(args.destdir)
    if not destdir.exists():
        print(f'Destination directory {destdir} does not exist. creating it...')
        destdir.mkdir(parents=True, exist_ok=True)
        print(f'Directory {destdir} created.')
    else: 
        if not destdir.is_dir():
            raise ValueError(f'Destination path {destdir} is not a directory.')
    
    # ROOT DIR
    rootdir = Path(args.rootdir)
    if not rootdir.exists():
        raise ValueError(f'Root directory {rootdir} does not exist.')
    elif not rootdir.is_dir():
        raise ValueError(f'Root path {rootdir} is not a directory.')
    #check for data
    root_glob = ''
    l1c_files = sorted(rootdir.glob('*l1c*.nc'))
    if len(l1c_files) < 1: #no.nc files in rootdir, check if subdirs havee .nc files
        root_glob = '**/'
        l1c_files = sorted(rootdir.glob(root_glob + '*l1c*.nc'))
        if len(l1c_files) < 1: #no .nc files in subdirs either
            raise ValueError(f'No L1C files found in root directory {rootdir} or subdirectories.')
        else:... #if subdirs needed, make the list here

    # SOLAR DIR
    if args.solardir is None:
        print('No solar directory provided. No daytime data will be processed.')
    else: #string is provided, check it
        soldir = Path(args.soldir)
        if not soldir.exists(): #does it exists?
            raise ValueError(f'Solar directory {soldir} does not exist.')
        elif not soldir.is_dir(): #is it a dir?
            raise ValueError(f'Solar path {soldir} is not a directory.')
        else: #exits and is dir, check for .nc files
            solar_glob = ''
            solar_files = sorted(soldir.glob(solar_glob'*l1c*.nc'))
            if len(solar_files) < 1: #no .nc files in rootdir, check if subdirs havee .nc files
                solar_glob = '**/'
                solar_files = sorted(soldir.glob(solar_glob+'*l1c*.nc'))
                if len(solar_files) < 1: #no .nc files in subdirs either
                    raise ValueError(f'No Solar L1C files found in solar directory {soldir} or subdirectories.')
                else: ... #if subdirs needed, make the list here
    
    ############## PROCESSING PARAMETERS ##############
    
    # WINDOWS TO PROCESS
    if args.windows is None:
        print('No windows provided. Processing all available windows.')
        valid_windows = np.unique([get_win_from_fn(f) for f in l1c_files])
    else:
        # check if win exists in data
        valid_windows = [w for w in args.windows for w in args.windows if len(list(rootdir.glob(rootdir + f'*{w}*.nc' ))) > 0]  #type: ignore
        # check if win exists in sol data
        if args.soldir is not None:
            valid_windows = [w for w in valid_windows if len(list(soldir.glob(soldir + f'*{w}*.nc' ))) > 0]  #type: ignore
        if len(valid_windows) < 1:
            raise ValueError(f'None of the provided windows {args.windows} exist in the data.')
    
    args.windows = valid_windows
    print(f'Processing windows: {args.windows}')

    # DATES TO PROCESS
    all_valid_dates = np.unique([get_date_from_fn(f) for f in l1c_files]) 
    if args.dates is None:
        print('No dates provided. Processing all available dates.')
        print(f'Processing dates: {all_valid_dates}')
        args.dates = sorted(all_valid_dates)
    else:
        valid_dates  = [d for d in args.dates if d in all_valid_dates]
        if len(valid_dates) < 1:
            raise ValueError(f'None of the provided dates {args.dates} exist in the data.')
        args.dates = sorted(valid_dates) #reassign to args for later use
        print(f'Processing dates: {args.dates}')


    # DESTINATION PREFIX
    # if no prefix provided, use rootdir name 
    if args.dest_prefix is None:
        dest_prefix = 'hmsao-l2'
        print(f'No destination prefix provided. Using default prefix: {dest_prefix}')
    else:
        dest_prefix = args.dest_prefix
        if 'l2' not in args.dest_prefix.lower():
            dest_prefix += '_l2'
    args.dest_prefix = dest_prefix #reassign to args for later use

    # ZENITH ANGLE BINSIZE 
    if args.zabinsize is None:
        args.zabinsize = 1.5
    else:
        args.zabinsize = float(args.zabinsize)


    
    #run function for each window
    ...







