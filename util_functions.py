#%%
import csv
import numpy as np
import xarray as xr
from typing import Iterable



def get_bounds_from_csv(csv_path: str, wavelength: str):
    """
    Get the bounds from the csv file.
    """
    with open(csv_path) as f:
        bg_dict = {}
        feature_dict = {}
        za_dict = {}
        for row in csv.reader(f):
            key = row[0]
            if key.startswith('#'):
                continue
            elif key.startswith(wavelength) and 'bg' in key:
                bg_dict[key] = slice(float(row[1]), float(row[2]))
            elif key.startswith(wavelength) and 'za' in key:
                za_dict[key] = slice(float(row[1]), float(row[2]))
            elif key.startswith(wavelength):
                feature_dict[key] = slice(float(row[1]), float(row[2]))
        if len(bg_dict) == 0 and len(feature_dict) == 0:
            raise ValueError(f'No bounds found for {wavelength} in {csv_path}')
    return feature_dict, bg_dict, za_dict


def str2bool(value: str) -> bool:
    if value.lower() in ('true', '1', 't', 'y', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'f', 'n', 'no'):
        return False
    raise ValueError("Invalid boolean value: {}".format(value))


def rms_func(data, axis=None):
    return np.sqrt(np.sum(data**2, axis=axis))

def align_spectra_core(sol: np.ndarray, sky: np.ndarray, wavelength: np.ndarray, offset: float = None) -> np.ndarray:
    sol_ = np.nan_to_num(sol)
    sky_ = np.nan_to_num(sky)

    if offset is None: #automatically find a lag by using scipy.signal.correlate
        sky_norm = (sky_ - np.mean(sky_)) / np.std(sky_)
        sol_norm = (sol_ - np.mean(sol_)) / np.std(sol_)
        corr = np.correlate(sky_norm, sol_norm, mode='full')
        lag = np.arange(-len(sol_) + 1, len(sky_))[np.argmax(corr)]
        offset = -lag * np.mean(np.diff(wavelength))

    shifted_wl = wavelength - offset
    interp_sol = np.interp(wavelength, shifted_wl, sol_, left=np.nan, right=np.nan)
    return interp_sol

def apply_alignment(sol_ds: xr.DataArray, sky_ds: xr.DataArray, offset:float = None) -> xr.DataArray:
    wl = sky_ds["wavelength"]
    # sky_ds = sky_ds.drop_vars('tstamp')
    # sol_ds = sol_ds.drop_vars('tstamp')

    # sky_ds,sol_ds = xr.broadcast(sky_ds, sol_ds)
    aligned = xr.apply_ufunc(
        align_spectra_core,
        sol_ds['intensity'],
        sky_ds['intensity'],
        wl,
        input_core_dims=[["wavelength"], ["wavelength"], ["wavelength"]],
        output_core_dims=[["wavelength"]],
        vectorize=True,
        dask="parallelized",
        kwargs={"offset": offset},
        on_missing_core_dim="copy",
        dask_gufunc_kwargs = {'allow_rechunk':True}
        

    )
    return aligned

def Residual(sf:int,skyspec:Iterable[float],solspec:Iterable[float])-> Iterable[float]:return solspec - sf*skyspec # type: ignore


def solar_subtraction_core(sky:np.ndarray, sol:np.ndarray, wavelength:np.ndarray )->np.ndarray:
    mask = np.isfinite(sky) & np.isfinite(sol)
    # print(f'sky:{sky}, sol:{sol}')
    mask = np.isfinite(sky) & np.isfinite(sol)

    # print(mask)
    if len(mask) <1 :
        sky_masked = sky
        sol_masked = sol
        wavelength_masked = wavelength
    else:
        sky_masked = sky[mask]
        sol_masked = sol[mask]
        wavelength_masked = wavelength[mask]
    n = 3
    leftmask =np.arange(0,30,n)
    rightmask = np.append(np.arange(-30,-1,n),-1)
    
    sf1 = sky_masked[leftmask]/sol_masked[leftmask]
    sf2 = sky_masked[rightmask]/sol_masked[rightmask]
    rsf = np.concatenate((sf1, sf2))
    rwl = np.concatenate((wavelength[leftmask], wavelength[rightmask]))
    sf = np.interp(wavelength_masked, rwl, rsf)
    sf_inverse = [1/s for s in sf]
    sf_inverse = np.array(sf_inverse)
    scaled_sky = np.asanyarray(sky_masked.copy()) * sf_inverse
    res = scaled_sky - sol_masked

    res_full = np.full_like(sky, np.nan)
    res_full[mask] = res
    return res_full
    


def apply_solar_subtraction(sol_ds: xr.Dataset, sky_ds: xr.Dataset):

    sol_da = sol_ds['intensity'] if isinstance(sol_ds, xr.Dataset) else sol_ds
    sky_da = sky_ds['intensity'] if isinstance(sky_ds, xr.Dataset) else sky_ds

    # Broadcast sol to sky (sol is za,wavelength)
    sol_da = sol_da.expand_dims(tstamp=sky_da.tstamp)

    # wavelength = sky_da.coords["wavelength"].values  # FULL array
    subtracted = xr.apply_ufunc(
    solar_subtraction_core,
    sky_da,
    sol_da,
    sky_da.wavelength,
    input_core_dims=[['wavelength'], ['wavelength'], ['wavelength']],
    output_core_dims=[['wavelength']],
    vectorize=True,
    dask="parallelized",
    output_dtypes=[sky_da.dtype],
    dask_gufunc_kwargs={'allow_rechunk': True},
)

    return subtracted

def find_outlier_pixels(data, tolerance=3, worry_about_edges=True):
    # This function finds the hot or dead pixels in a 2D dataset.
    # tolerance is the number of standard deviations used to cutoff the hot pixels
    # If you want to ignore the edges and greatly speed up the code, then set
    # worry_about_edges to False.
    #
    # The function returns a list of hot pixels and also an image with with hot pixels removed

    from scipy.ndimage import median_filter
    blurred = median_filter(data, size=2)
    difference = data - blurred
    threshold = tolerance*np.std(difference)

    # find the hot pixels, but ignore the edges
    hot_pixels = np.nonzero((np.abs(difference[1:-1, 1:-1]) > threshold))
    # because we ignored the first row and first column
    hot_pixels = np.array(hot_pixels) + 1

    # This is the image with the hot pixels removed
    fixed_image = np.copy(data)
    for y, x in zip(hot_pixels[0], hot_pixels[1]):
        fixed_image[y, x] = blurred[y, x]

    if worry_about_edges == True:
        height, width = np.shape(data)

        ### Now get the pixels on the edges (but not the corners)###

        # left and right sides
        for index in range(1, height-1):
            # left side:
            med = np.median(data[index-1:index+2, 0:2])
            diff = np.abs(data[index, 0] - med)
            if diff > threshold:
                hot_pixels = np.hstack((hot_pixels, [[index], [0]]))
                fixed_image[index, 0] = med

            # right side:
            med = np.median(data[index-1:index+2, -2:])
            diff = np.abs(data[index, -1] - med)
            if diff > threshold:
                hot_pixels = np.hstack((hot_pixels, [[index], [width-1]]))
                fixed_image[index, -1] = med

        # Then the top and bottom
        for index in range(1, width-1):
            # bottom:
            med = np.median(data[0:2, index-1:index+2])
            diff = np.abs(data[0, index] - med)
            if diff > threshold:
                hot_pixels = np.hstack((hot_pixels, [[0], [index]]))
                fixed_image[0, index] = med

            # top:
            med = np.median(data[-2:, index-1:index+2])
            diff = np.abs(data[-1, index] - med)
            if diff > threshold:
                hot_pixels = np.hstack((hot_pixels, [[height-1], [index]]))
                fixed_image[-1, index] = med
        ### Then the corners###

        # bottom left
        med = np.median(data[0:2, 0:2])
        diff = np.abs(data[0, 0] - med)
        if diff > threshold:
            hot_pixels = np.hstack((hot_pixels, [[0], [0]]))
            fixed_image[0, 0] = med

        # bottom right
        med = np.median(data[0:2, -2:])
        diff = np.abs(data[0, -1] - med)
        if diff > threshold:
            hot_pixels = np.hstack((hot_pixels, [[0], [width-1]]))
            fixed_image[0, -1] = med

        # top left
        med = np.median(data[-2:, 0:2])
        diff = np.abs(data[-1, 0] - med)
        if diff > threshold:
            hot_pixels = np.hstack((hot_pixels, [[height-1], [0]]))
            fixed_image[-1, 0] = med

        # top right
        med = np.median(data[-2:, -2:])
        diff = np.abs(data[-1, -1] - med)
        if diff > threshold:
            hot_pixels = np.hstack((hot_pixels, [[height-1], [width-1]]))
            fixed_image[-1, -1] = med

    return hot_pixels, fixed_image