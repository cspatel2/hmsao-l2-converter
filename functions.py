#%%
import csv
import numpy as np
import xarray as xr
from typing import Iterable
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

def find_feature_width_core(data:np.ndarray,wavelength:np.ndarray, prominence:float=0.5, rel_height:float=0.7) -> tuple[float | None, float | None]:
    """ Finds feature boundaries using scipy.signal.find_peaks() and scipy.signal.peak_widths().

    Args:
        data (np.ndarray): 1D array of spectral data of shape (n,).
        wavelength (np.ndarray): 1D array of corresponding wavelengths. Same shape as data (n,).
        prominence (float, optional): the required prominence of peaks. Defaults to 0.5.
        rel_height (float, optional): Chooses the relative height at which the peak width is measured as a percentage of its prominence. 1.0 calculates the width of the peak at its lowest contour line while 0.5 evaluates at half the prominence height. Must be at least 0. Defaults to 0.7.

    Returns:
        tuple[float | None, float | None]: _description_
    """    
    lf = data.copy()
    lf -= np.nanmin(lf)
    lf /= np.nanmax(lf)
    lf = gaussian_filter1d(lf, sigma=3)
    peaks,_ = find_peaks(lf, prominence=prominence)
    len(peaks)
    if len(peaks) == 0:
        return None, None
    widths, height,left,right = peak_widths(lf, peaks, rel_height=rel_height)
    return float(wavelength[round(left[0])]), float(wavelength[round(right[0])]) 
#%%
def get_feature_bounds(win:str, da:xr.DataArray, bgoffset:float=0.01, prominence:float=0.5, rel_height:float=0.7, returnfullds:bool=False) -> xr.Dataset | dict[str,  float]:
    """ find the min and max wavelength boundaries of the feature of interest and its that of its corresponding background(s).

    Args:
        win (str): window name or identifier for the feature.
        da (xr.DataArray): _dataarray containing the spectral data with 'wavelength' dimension.
        bgoffset (float, optional): the distance between feature and background windows in nm. Defaults to 0.01.
        prominence (float, optional): the required prominence of peaks. Defaults to 0.5.
        rel_height (float, optional): the relative height at which to calculate peak widths. Defaults to 0.7.
        returnfullds (bool, optional): whether to return full datasets of bounds or single values. Defaults to False. if true, each boundary will be an array as a function of za. If false, single min/max values will be returned.

    Returns:
        xr.Dataset | dict[str,  float]: dataset or dictionary containing feature and background wavelength bounds. (see returnfullds description for description of outputs)
    """    
    l_start, l_stop = xr.apply_ufunc(
        find_feature_width_core,
        da,
        da.wavelength,
        input_core_dims=[['wavelength'], ['wavelength']],
        output_core_dims=[[], []],      # TWO scalar outputs
        vectorize=True,
        dask='parallelized',
        kwargs={'prominence': prominence, 'rel_height': rel_height}
    )
    l_start, l_stop = l_start.data, l_stop.data
    b1_end = l_start - bgoffset
    b1_start = b1_end - np.abs(l_stop - l_start)
    b2_start = l_stop + bgoffset
    b2_end = b2_start + np.abs(l_stop - l_start)

    if returnfullds:
        boundsds = xr.Dataset(
            data_vars={
                f'{win}': (('za','edge'), np.array([l_start, l_stop]).T, {'long_name':'feature wavelength', 'units':'nm'}),
                f'b1_{win}': (('za','edge'), np.array([b1_start, b1_end]).T, {'long_name':'background 1 wavelength', 'units':'nm'}),
                f'b2_{win}': (('za','edge'), np.array([b2_start, b2_end]).T, {'long_name':'background 2 wavelength', 'units':'nm'}),
            },
            coords={'za': (('za'), 
                        da.za.data, 
                        da.za.attrs
                        ),
                    'edge': (('edge'), 
                            ['start', 'stop'], 
                            {'description':'start = start wavelength; stop = stop wavelength (end)','note':'start < stop by definition'})}
            
        )
        return boundsds
    else:
        start = float(np.nanmin(l_start))
        stop = float(np.nanmax(l_stop))
        width = np.abs(stop - start)
        return {
            'start': start,
            'stop': stop,
            'b1_start': float(start - width - bgoffset),
            'b1_end': float(stop - width - bgoffset), 
            'b2_start': float(stop + bgoffset),
            'b2_end': float(stop + width + bgoffset)
        }