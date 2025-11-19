#%%
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pathlib import Path    
# %%
ds = xr.open_dataset(list(Path('').glob('*.nc'))[0])
# %%
ds
# %%
win = '5577'
