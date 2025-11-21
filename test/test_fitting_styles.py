#%%
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pathlib import Path

from yaml import warnings    
from functions import get_bounds_from_csv
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths
# %%
win = '5577'
datadir = Path('../data/l1c')
fns = np.sort(list(datadir.glob(f'*{win}*.nc'))) # type:ignore
ds = xr.open_dataset(fns[0])
# %%git 
# wlslice = slice(557.4, 558)
wlslice = slice(None, None)
#%%
bounds, bgbounds, zabounds = get_bounds_from_csv('bounds.csv', win)
# %%
xmin = 557.6
xmax = 557.8
lslice = slice(xmin, xmax)

bg1_min = xmin - 0.1 - (xmax - xmin)  
bg1_max = xmin - 0.1 
bg1_slice = slice(bg1_min, bg1_max)

bg2_min = xmax + 0.1
bg2_max = xmax + 0.1 + (xmax - xmin)
bg2_slice = slice(bg2_min, bg2_max)
# %%
da = ds.intensity.isel(tstamp = 0).sel(wavelength=wlslice)
dvmin = np.nanpercentile(da, 1)
dvmax = np.nanpercentile(da, 99)
da.plot(vmin=dvmin, vmax=dvmax)
plt.axvline(lslice.start, color='red')
plt.axvline(lslice.stop, color='red')
plt.axvline(bg1_slice.start, color='green')
plt.axvline(bg1_slice.stop, color='green')
plt.axvline(bg2_slice.start, color='green')
plt.axvline(bg2_slice.stop, color='green')
#%%
line = ds.intensity.sum('tstamp').isel(za = -30)

lf = line.copy()
lf.data -= np.nanmin(lf.data)
lf.data /= np.nanmax(lf.data)

lf.plot(lw = 0.5)
lf.data = gaussian_filter1d(lf.data, sigma=3)
lf.plot(lw = 0.8)
peaks,_ = find_peaks(lf.data, prominence=0.5)
plt.plot(lf.wavelength.data[peaks], lf.data[peaks], "x")
widths, height,left,right = peak_widths(lf.data, peaks, rel_height=0.7)
plt.axvline(lf.wavelength.data[int(left[0])], color='red', lw = 0.5)
plt.axvline(lf.wavelength.data[int(right[0])], color='red', lw = 0.5)

# %%
def find_feature_width_core(data:np.ndarray,wavelength:np.ndarray, prominence:float=0.5, rel_height:float=0.7):
    lf = data.copy()
    lf -= np.nanmin(lf)
    lf /= np.nanmax(lf)
    lf = gaussian_filter1d(lf, sigma=3)
    peaks,_ = find_peaks(lf, prominence=prominence)
    len(peaks)
    if len(peaks) == 0:
        return None, None
    widths, height,left,right = peak_widths(lf, peaks, rel_height=rel_height)
    return wavelength[round(left[0])], wavelength[round(right[0])] 
#%%
def get_feature_bounds(da:xr.DataArray, bgoffset:float=0.1, prominence:float=0.5, rel_height:float=0.7, returnfullds:bool=False) -> xr.Dataset | dict[str,  float]:

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
            'b1_start': float(start - width - bgoffset),  #float(np.nanmin(b1_start)),
            'b1_end': float(stop - width - bgoffset),    #float(np.nanmax(b1_end)),
            'b2_start': float(stop + bgoffset),
            'b2_end': float(stop + width + bgoffset)
        }



    

#%%

# bdict = get_feature_bounds(da, bgoffset=0.05, prominence=0.5, rel_height=0.75, returnfullds=False)
bds = get_feature_bounds(da, bgoffset=0.05, prominence=0.5, rel_height=0.75, returnfullds=True)

#%%    
da.plot(vmin=dvmin, vmax=dvmax)
plt.axvline(bdict['start'], color='red')  #type:ignore
plt.axvline(bdict['stop'], color='red') #type:ignore
plt.axvline(bdict['b1_start'], color='green') #type:ignore
plt.axvline(bdict['b1_end'], color='green') #type:ignore
plt.axvline(bdict['b2_start'], color='green') #type:ignore
plt.axvline(bdict['b2_end'], color='green') #type:ignore

# %%
def peakShirley(x,data):
    bck = np.zeros(data.shape)
    bck_old = np.zeros(data.shape)
    for j in range(10):
        k = (data[0]-data[-1])/np.trapezoid(data-data[-1]-bck_old,x)
        for i in range(len(x)):
            bck[i] = k*np.trapezoid(data[i:]-data[-1]-bck_old[i:],x[i:])
        bck_old = bck
    background = bck + data[-1]
    signal = data - background
    plt.plot(x,data, label = 'data', alpha=0.5)
    plt.plot(x,background, label = 'background')
    plt.plot(x,signal, label = 'signal')
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.plot(x, ((signal+background)/background)*1.e7, label='SNR * 1e7')

    plt.legend()
    plt.show()
    return signal,background
    # return bck+data[-1]
    #return np.zeros(sig.shape,dtype=float)+(sig[0]+sig[-1])*0.5
# %%

# vals = [val for k,val in bdict.items()]
# wlslice = slice(np.min(vals), np.max(vals))
offset = 0.001
wlslice = slice(bdict['start']-offset, bdict['stop']+offset)
line = ds.intensity.isel(tstamp = 10,za = -1).sel(wavelength=wlslice)

res = peakShirley(line.wavelength.data, line.data)
# %%
line.data[-1]
# %%

#%%

# %%
def _calculate_shirley_background_full_range(
    xps: np.ndarray, eps=1e-7, max_iters=50, n_samples=5
) -> np.ndarray:
    """Core routine for calculating a Shirley background on np.ndarray data."""
    background = np.copy(xps)
    cumulative_xps = np.cumsum(xps, axis=0)
    total_xps = np.sum(xps, axis=0)

    rel_error = np.inf

    i_left = np.mean(xps[:n_samples], axis=0)
    i_right = np.mean(xps[-n_samples:], axis=0)

    iter_count = 0

    k = i_left - i_right
    for iter_count in range(max_iters):
        cumulative_background = np.cumsum(background, axis=0)
        total_background = np.sum(background, axis=0)

        new_bkg = np.copy(background)

        for i in range(len(new_bkg)):
            new_bkg[i] = i_right + k * (
                (total_xps - cumulative_xps[i] - (total_background - cumulative_background[i]))
                / (total_xps - total_background + 1e-5)
            )

        rel_error = np.abs(np.sum(new_bkg, axis=0) - total_background) / (total_background)

        background = new_bkg

        if np.any(rel_error < eps):
            break

    if (iter_count + 1) == max_iters:
        return np.full(background.shape, np.nan)
        # raise Warning(
        #     "Shirley background calculation did not converge "
        #     + "after {} steps with relative error {}!".format(max_iters, rel_error)
        # )
    return background
# %%
offset = 0.00
wlslice = slice(bdict['start']-offset, bdict['stop']+offset)
line = ds.intensity.isel(tstamp = 10,za =-10).sel(wavelength=wlslice)
bg = _calculate_shirley_background_full_range(line.data, eps=1e-7, max_iters=150, n_samples=20)

plt.plot(line.wavelength.data, line.data, label='data', alpha=0.5)
plt.plot(line.wavelength.data, bg, label='background')
plt.plot(line.wavelength.data, line.data - bg, label='signal')
plt.axhline(0, color='black', lw=0.5, ls='--')
plt.legend()
# %%
#%%


def shirley_baseline(x, y, max_iter=50, tol=1e-8):
    """
    Compute Shirley background using explicit cumulative integral (manual trapz).
    
    Parameters
    ----------
    x : array_like
        1-D array of x-values (wavelengths or energies)
    y : array_like
        1-D array of signal intensities
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
    
    Returns
    -------
    B : ndarray
        Shirley background, same shape as y
    """
    y = np.array(y, dtype=float)
    B = np.zeros_like(y)
    B_old = np.ones_like(y)  # initialize with something different
    y_end = y[-1]
    
    for _ in range(max_iter):
        # constant k from boundary conditions
        denom = np.trapezoid(y - y_end - B_old, x)
        if denom == 0:
            return np.zeros_like(y)
        k = (y[0] - y_end) / denom

        # manual cumulative integral from i -> end
        integral = np.zeros_like(y)
        for i in range(len(y)):
            integral[i] = np.trapezoid(y[i:] - y_end - B_old[i:], x[i:])

        B = k * integral
        
        # check convergence
        if np.max(np.abs(B - B_old)) < tol:
            break
        B_old = B.copy()
    
    return B + y_end
#%%
from scipy import sparse
from scipy.sparse.linalg import spsolve
def baseline_als(y, lam, p, niter=10):
    """
    Asymmetric Least Squares Smoothing for baseline correction.

    Parameters:
    -----------
    y : array-like
        The input signal (e.g., spectrum).
    lam : float
        Smoothness parameter (lambda). Larger values make the baseline stiffer.
    p : float
        Asymmetry parameter. Controls the asymmetry of the weights.
        Typically between 0.001 and 0.1.
    niter : int, optional
        Number of iterations. Default is 10.

    Returns:
    --------
    z : array-like
        The estimated baseline.
    """
    L = len(y)
    # Create a sparse matrix for the second derivative
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)  # Initial weights
    z = np.nan 
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        # Solve (W + lambda * D.T @ D) * z = W * y
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        # Update weights based on the difference between signal and baseline
        w = p * (y > z) + (1 - p) * (y < z)
    return z 

# %%

def compare_baselines(x, y,
                      als_lam=1e5, als_p=0.01,
                      als_niter=40,
                      shirley_iter=50):

    # Compute baselines
    b_shirley = shirley_baseline(x, y, max_iter=shirley_iter)
    b_als      = baseline_als(y, lam=als_lam, p=als_p, niter=als_niter)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, label='Original spectrum', color='k', lw=1)

    plt.plot(x, b_shirley, label='Shirley baseline', lw=2)
    plt.plot(x, b_als, label='ALS baseline', lw=2) # type: ignore

    plt.legend()
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity (Rayleigh/nm)')
    plt.title('Baseline Comparison: Shirley vs ALS')
    plt.show()

    return b_shirley, b_als
# %%
offset = 0.0
zidx = -30
wlslice = slice(bds[f'{win}'].sel(edge='start').isel(za = zidx).data - offset,bds[f'{win}'].sel(edge='stop').isel(za = zidx).data + offset)
# wlslice = slice(bds['start']-offset, bds['stop']+offset)
line = ds.intensity.isel(tstamp = 10,za =zidx).sel(wavelength=wlslice)
bs,ba = compare_baselines(line.wavelength.data, line.data,
                  als_lam=1e3, als_p=0.001, als_niter=100,
                  shirley_iter=20)
# %%
