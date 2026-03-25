# %%
from __future__ import annotations
from datetime import datetime, timezone, timedelta
from arrow import get
import astropy.units as u
from astropy.coordinates import EarthLocation, AltAz
from astropy.time import Time
from astropy.coordinates import get_sun, SkyCoord
import matplotlib.pyplot as plt
import numpy as np
from typing import Iterable, SupportsFloat as Numeric
from pytz import UTC
from pysolar.solar import get_altitude
from suncalc import get_position, get_times


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
    test_date = datetime(2025, 11, 17, tzinfo=timezone.utc) 
    res = [test_date]
    res = [test_date + timedelta(hours=i) for i in range(-12, 13)]
    tstamps = [r.timestamp() for r in res]
    sza = [solar_zenith_angle(t, latitude,longitude,elevation) for t in tstamps]
    sza_pysolar = [
        90 - get_altitude(latitude, longitude, r, elevation=elevation) for r in res
    ]
    sza_suncalc = [
        90 - get_position(r, longitude, latitude)['altitude']*180/np.pi for r in res
    ]
    dawn = get_times(test_date, longitude, latitude)['nautical_dawn'] # today
    dusk = get_times(test_date - timedelta(days=1), longitude, latitude)['nautical_dusk'] # previous day


    # altitude_deg = [get_altitude(latitude, longitude, r) for r in res]
    # zenith_angle_deg = 90 - np.asarray(altitude_deg)


    plt.plot(res,sza, ls=':') # type: ignore
    plt.plot(res,sza_pysolar, ls='--') # type: ignore
    plt.plot(res,sza_suncalc, ls='-.') # type: ignore
    # plt.plot(res, zenith_angle_deg, '--') # type: ignore
    plt.axhline(90, ls='-', color='k')
    plt.gcf().autofmt_xdate()
    plt.xlabel('time (UTC)')
    plt.ylabel('Solar Zenith Angle (deg)')
    plt.title('Location: Kiruna, Sweden')
    plt.axvline(dusk, ls='--', color='orange', label='nautical dusk')
    plt.axvline(dawn, ls='--', color='cyan', label='nautical dawn')
    plt.axhline(112, ls='--', color='red', label='SZA=112°')
    plt.axhline(96, ls='--', color='gray', label='SZA=96°')
    plt.legend()
    # plt.ylim(180, 0)
    # plt.ylim(np.max(sza)+5, np.min(sza)-5)
    plt.show()
    
#%%
