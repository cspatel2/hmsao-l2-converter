#%%
from sqlite3.dbapi2 import Timestamp

import numpy as np
from prometheus_client import instance_ip_grouping_key
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, List, Optional, Self, Tuple, Union, overload, get_args
from typing_extensions import Literal
from dataclasses import asdict, dataclass, field, fields
from scipy.optimize import curve_fit
#%%
InstrVersion = Literal["v1", "v2"]
DataLevel = Literal["l1a", "l1b", "l1c"]

@dataclass
class DataInfo:
    p: Union[Path, str]
    kind: InstrVersion
    level: str
    window: str
    date: str

    def __post_init__(self):
        if len(self.date) != 8:
            raise ValueError(f"Date must be in 'YYYYMMDD' format, got {self.date}.")
        if self.level not in get_args(DataLevel):
            raise ValueError(
                f"Invalid level: {self.level}. Must be one of {get_args(DataLevel)}."
            )
        if self.kind not in get_args(InstrVersion):
            raise ValueError(
                f"Invalid kind: {self.kind}. Must be one of {get_args(InstrVersion)}."
            )
    def _get_files(self) -> List[Path]:
        level = self.level
        version = self.kind
        win = self.window
        if isinstance(self.p, str):
            p = Path(self.p).expanduser()
        else:
            p = self.p.expanduser()

        if level == "l1b" or level == "l1a":
            fns = list(p.glob(f"*{version}/{level}/*/*{self.date}*{win}*.nc"))
        elif level == "l1c":
            fns = list(p.glob(f"*{version}/l1c/*{self.date}*{win}*.nc"))
        else:
            raise ValueError(f"Invalid level: {level}. Must be 'l1a', 'l1b' or 'l1c'.")
        fns.sort()
        return fns
    
    def replace_version_in_path(self, path: Path, new_level: str) -> Path:
        fn = path
        name = fn.stem.split('[')[0]
        nfn = fn.parent /( name + '.nc')
        nfn = Path(str(nfn).replace(f"{self.level}", f"{new_level}"))
        return nfn


    def load(
        self,
        tidx: Optional[Union[datetime, int]] = None,
        za_bin: Optional[int] = None,
        zaidx: Optional[Union[int, float]] = None,
        plot: bool = False,
        plotsave: bool = False,
    ) -> xr.Dataset:
        fns = self._get_files()
        ds = xr.open_mfdataset(fns, combine="by_coords")
        ds = ds.assign_coords(
            {
                "time": (
                    "tstamp",
                    [
                        datetime.fromtimestamp(t, tz=timezone.utc)
                        for t in ds["tstamp"].values
                    ],
                )
            }
        )
        if "param" in list(ds.variables):
            ds = ds.drop_vars("param")


        if tidx := tidx:
            if isinstance(tidx, int):
                ds = ds.isel(time=tidx)
            elif isinstance(tidx, datetime):
                ds = ds.sel(time=tidx, method="nearest")
            else:
                raise ValueError(
                    f"Invalid tidx: {tidx}. Must be an integer index or a datetime object."
                )
        if za_bin := za_bin:
            ds = ds.coarsen(za=za_bin, boundary="trim").sum()  # type: ignore
        ds_plot = ds.copy()
        if zaidx := zaidx:
            if isinstance(zaidx, int):
                ds = ds.isel(za=zaidx)
            elif isinstance(zaidx, float):
                ds = ds.sel(za=zaidx, method="nearest")
            else:
                raise ValueError(
                    f"Invalid zaidx: {zaidx}. Must be an integer index or a float value for nearest selection."
                )
        if plot:
            plt.figure(figsize=(6.4, 4.8), dpi=300)
            id = "countrate" if "countrate" in list(ds.data_vars) else "intensity"

            ds_plot[id].plot() # type: ignore
            plt.axhline(float(ds.za.values), color="red", lw=1, ls="--")
            plt.xlabel(r"Wavelength [$nm$]")
            plt.ylabel(r"Zenith Angle [$^\circ$]")
            dt = datetime.fromtimestamp(float(ds.tstamp.values), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            plt.title(f"{dt}")
            plt.show()
            if plotsave:
                savedir = Path("plots")
                savedir.mkdir(exist_ok=True)
                savepath = savedir / f"data_overview_{self.date}_{win}_{zaidx}.png"
                plt.savefig(savepath, dpi=300, bbox_inches="tight")
            
        return ds


def Gaussian(x,xo, a, w):
    return a*np.exp(-(x-xo)**2/(2*w**2))
def ConstantBackground(x, bg):
    return np.full_like(x, bg)

def fit_function(x, xo, a, w, bg):
    return Gaussian(x, xo, a, w) + ConstantBackground(x, bg)


def fit_spectral_line_1d_core(x:np.ndarray, y:np.ndarray, yerr:np.ndarray, p0:dict):
    p_arr = [val for i,val in p0.items()]
    try:
        popt, pcov = curve_fit(
                fit_function,
                x,
                y,
                p0=p_arr,
                sigma=yerr,
                absolute_sigma=True,
            )

        return np.concatenate([popt, pcov.ravel()]) 
    except RuntimeError as e:
        # print(f"Fit failed for data with shape {y.shape}: {e}")
        return np.full(len(p0)+len(p0)**2, np.nan)  # Return NaNs if fit fails
    
def fit_spectral_line(da_data: xr.DataArray, da_err: xr.DataArray, p0: list, t_chunksize = 3):
    da_full = da_data.chunk(
        {
            "wavelength": -1,  # MUST be -1 (one chunk)
            "za": -1,  # Keeps spatial columns intact
            "tstamp": t_chunksize,  # Parallelizes the fit across groups of 20 images
        }
    )
    da_err_full = da_err.chunk(
        {
            "wavelength": -1,  # MUST be -1 (one chunk)
            "za": -1,  # Keeps spatial columns intact
            "tstamp": t_chunksize,  # Parallelizes the fit across groups of 20 images
        }
    )
    wl_arr = da_data.wavelength.values

    fit = xr.apply_ufunc(
        fit_spectral_line_1d_core,
        wl_arr,
        da_full,
        da_err_full,
        input_core_dims=[["wavelength"], ["wavelength"], ["wavelength"]],
        output_core_dims=[["fit_param"]],
        kwargs={"p0": p0},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"fit_param": len(p0)+len(p0)**2}},  # 4 fit parameters + 16 covariance elements
    )

    # fit values
    xo_arr = fit.isel(fit_param=0)
    a_arr = fit.isel(fit_param=1)
    w_arr = fit.isel(fit_param=2)
    bg_arr = fit.isel(fit_param=3)

    #errors
    a_err = np.sqrt(fit.isel(fit_param=4 + 5))  # row 1, col 1
    w_err = np.sqrt(fit.isel(fit_param=4 + 10))  # row 2, col 2
    aw_err = fit.isel(fit_param=4 + 6)  # covariance between a and w


    # area under the Gaussian curve is a * |w| * sqrt(2*pi)
    line_brightness = a_arr * np.abs(w_arr) * np.sqrt(2 * np.pi)

    #error
    deriv_a = np.abs(w_arr) * np.sqrt(2 * np.pi)
    deriv_w = a_arr * np.sqrt(2 * np.pi)
    line_error = np.sqrt((deriv_a * a_err)**2 + (deriv_w * w_err)**2 + 2 * deriv_a * deriv_w * aw_err)

    # print(np.shape(line_brightness), np.shape(line_error))
    #unit correction for ida and errda
    o_units = da_data.attrs.get("units", "")
    new_units = o_units.replace("/nm", "") if "/nm" in o_units else o_units.replace(".nm", "")

    new_coords = da_data.coords.copy()
    new_coords = new_coords.drop_dims('wavelength')
    ida = xr.DataArray(line_brightness, coords=new_coords, attrs={"units": new_units})
    errda = xr.DataArray(line_error, coords=new_coords, attrs={"units": new_units})


    return ida, errda
    
def calculate_optimal_time_chunks(
    da: xr.DataArray,
    time_dim: str = "tstamp",
    target_chunk_mb: float = 150.0,
) -> int:
    """Calculates the optimal number of frames for chunking along the time axis.

    Targeting 100MB-250MB per Dask chunk prevents scheduling overhead.

    Parameters
    ----------
    da : xr.DataArray
        The input data array containing the images (e.g., dss.intensity)
    time_dim : str
        The name of your time/image dimension (default is 'tstamp')
    target_chunk_mb : float
        Target memory size per chunk in Megabytes (default is 150.0 MB)

    Returns
    -------
    int
        Optimal chunk size (number of frames) to pass to t_chunksize.
    """
    # 1. Isolate one single 2D spatial-spectral slice (za x wavelength)
    # using .nbytes ensures it accounts for float32 vs float64 automatically
    single_frame_bytes = da.isel({time_dim: 0}).nbytes

    # 2. Convert target MB to bytes
    target_bytes = target_chunk_mb * 1024 * 1024

    # 3. Calculate how many frames comfortably fit into that memory footprint
    ideal_chunks = int(target_bytes / single_frame_bytes)

    # 4. Constrain bounds: Must be at least 1, and cannot exceed total time steps
    total_time_steps = da.sizes[time_dim]
    optimal_t_chunksize = max(1, min(ideal_chunks, total_time_steps))

    # print(f"--- Chunk Size Diagnostic ---")
    # print(f"Single frame size: {single_frame_bytes / 1024:.2f} KB")
    # print(f"Total time steps: {total_time_steps}")
    # print(f"Calculated optimal '{time_dim}' chunk size: {optimal_t_chunksize}")
    # print(
    #     f"Estimated memory per chunk: "
    #     f"{(single_frame_bytes * optimal_t_chunksize) / (1024*1024):.2f} MB"
    # )
    # print(f"-----------------------------")

    return optimal_t_chunksize

def convert_l1_to_l2(ds: xr.Dataset,
                     wlslice: slice | None = None,
                     tsslice: slice | None = None,
                     t_chunksize: int | None = None,
                     ) -> xr.Dataset:
    dss = ds.copy()
    dss = dss.drop_vars("time", errors="ignore")  # Drop 'time' dim if it exists, since we have 'tstamp' as coordinate
    if tsslice is not None:
        if isinstance(tsslice.start, datetime):
            tslice = slice(tsslice.start.timestamp(), tsslice.stop.timestamp())
            dss = dss.sel(tstamp = tslice)
        elif isinstance(tsslice.start, int):
            dss = dss.isel(tstamp=tsslice)
        elif isinstance(tsslice.start, float):
            tsslice = slice(tsslice.start, tsslice.stop)
            dss = dss.sel(tstamp=tsslice)
        else:
            raise ValueError(f"Invalid tsslice: {tsslice}. Must be a slice of datetime, int or float.")
        
    if wlslice is not None:
        if isinstance(wlslice.start, float):
            dss = dss.sel(wavelength=wlslice)
        elif isinstance(wlslice.start, int):
            dss = dss.isel(wavelength=wlslice)
        else:
            raise ValueError(f"Invalid wlslice: {wlslice}. Must be a slice of float or int.")

    if t_chunksize is None:
        t_chunksize = calculate_optimal_time_chunks(dss.intensity, time_dim="tstamp", target_chunk_mb=150.0)

    p0 = {
        "xo": cwl,
        "a": dss.intensity.max(skipna=True).values,
        "w": 0.1,
        "bg": dss.intensity.min(skipna=True).values
    }
    ida, errda = fit_spectral_line(dss.intensity, np.sqrt(dss.noise), p0, t_chunksize=t_chunksize)
    l2ds = xr.Dataset(
        {
            "brightness": ida,
            "noise": errda,
        }
    )
    new_attrs = {
        "description": "L2 dataset Gaussian fitted line brightness and associated noise (error) values.",
    }
    
    new_attrs = {
        "description": "L2 dataset Gaussian fitted line brightness and associated noise (error) values.",
        "ROI": dss.attrs.get("ROI", "Unknown"),
        "slit_size_um": dss.attrs.get("slit_size_um", "Unknown"),
        "DataProcessingLevel": "L2",
        "FileCreationDate": str(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")).encode('utf-8'),
        "ObservationLocation": dss.attrs.get("ObservationLocation", "Unknown")
    }
    l2ds.attrs  = new_attrs

    return l2ds


# %%
if __name__ == "__main__":
    info = DataInfo(
            p=Path("~/local_data_cedar"),
            kind="v2",
            level="l1c",
            window="5577",
            date="20251116",
        )
    fns = info._get_files()
    outfn = info.replace_version_in_path(fns[0], new_level="l2")
    outfn.parent.mkdir(exist_ok=True, parents=True)

    print(f"file to save to: {outfn.parent}")
    t_start = datetime(2025, 11, 16, 17, 0, tzinfo=timezone.utc)
    t_end = t_start + timedelta(hours= 7)
    tslice = slice(t_start.timestamp(), t_end.timestamp())
    print(f"start time: {t_start}\nend time: {t_end}")

    cwl = int(info.window)/10
    window_width = .5
    wlslice = slice(cwl - window_width/2, cwl + window_width/2)

    print(f"processing data...")
    ds = info.load(za_bin = None, tidx=None, zaidx=None, plot=False)
    nds = convert_l1_to_l2(ds, wlslice=wlslice, tsslice=tslice, t_chunksize=None) 
    
    print(f"saving to {outfn.stem}...")
    nds.to_netcdf(outfn)

#%%
# ds = xr.open_dataset('/home/charmi/local_data_cedar/hmsao-v2/l2/hmsaov2_l2_20251116_5577.nc')
# # %%
# da = ds.brightness * (4*np.pi * 1e-6) / 1e3
# da.attrs["units"] = "kR"
# vmin = np.nanpercentile(da.values, 1)
# vmax = np.nanpercentile(da.values, 99)
# plt.figure(figsize=(10, 3), dpi=300)
# da.plot(x = "tstamp", vmin=vmin, vmax=vmax, cmap= 'Greens')
# %%
