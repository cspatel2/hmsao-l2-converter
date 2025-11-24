# %%
from __future__ import annotations
from datetime import datetime, timezone, timedelta
import astropy.units as u
from astropy.coordinates import EarthLocation, AltAz
from astropy.time import Time
from astropy.coordinates import get_sun, SkyCoord
import matplotlib.pyplot as plt
import numpy as np
from typing import Iterable, SupportsFloat as Numeric
from pytz import UTC
from pysolar.solar import get_altitude


# %%

def solar_zenith_angle(tstamp:float, lat:float, lon:float, elevation:float)->float:
    """ Calcuates solar zenith angle using location and time.

    Args:
        tstamp (float): Seconds since UNIX epoch 1970-01-01 00:00:00 UTC
        lat (float): Latitude of observer in degrees. -90°(S) <= lat <= +90°(N) 
        lon (float): Longitude of observer in degrees.  Longitudes are measured increasing to the east, so west longitudes are negative.
        elevation (float): height in meters above sea level. 

    Returns:
        float: SZA range is  0° (Local Zenith) <= sza <= 180°
    """    
    # Create an EarthLocation object
    location:EarthLocation = EarthLocation(lon=lon*u.deg, lat=lat*u.deg, height=elevation*u.m) # type: ignore

    # Create a Time object from ttimestamp (in seconds since UNIX epoch) to UTC
    time = Time(tstamp, format='unix', scale='utc')

    # Create an AltAz frame
    altaz:AltAz = AltAz(obstime=time, location=location)

    # Get the Sun's position in AltAz coordinates
    sun:SkyCoord = get_sun(time)
    sun_altaz:SkyCoord = sun.transform_to(altaz, True)

    zenith:float = 90.0 - sun_altaz.alt.deg # type: ignore

    return zenith


if __name__ == '__main__':
    # Kiruna, Sweden coordinates
    longitude = 20.41
    latitude = 67.84 
    elevation = 420 # Approximate elevation
    print(type(datetime))
    test_date = datetime(2025, 3, 20, tzinfo=timezone.utc) 
    res = [test_date]
    for i in range(24*4):
        t = res[-1] + timedelta(minutes = 30)
        res.append(t)
    tstamps:Iterable[Numeric] = [r.timestamp() for r in res]
    sza:Iterable[Numeric] = [solar_zenith_angle(t, latitude,longitude,elevation) for t in tstamps]


    # altitude_deg = [get_altitude(latitude, longitude, r) for r in res]
    # zenith_angle_deg = 90 - np.asarray(altitude_deg)


    plt.plot(res,sza, ls=':') # type: ignore
    # plt.plot(res, zenith_angle_deg, '--') # type: ignore
    plt.axhline(90, ls='-', color='k')
    plt.gcf().autofmt_xdate()
    plt.xlabel('time (UTC)')
    plt.ylabel('Solar Zenith Angle (deg)')
    plt.title('Location: Kiruna, Sweden')
    plt.ylim(np.max(sza)+5, np.min(sza)-5)
    plt.show()
    
#%%
