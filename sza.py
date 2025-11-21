# %%

from datetime import datetime, timezone, timedelta
import astropy.units as u
from astropy.coordinates import EarthLocation, AltAz
from astropy.time import Time
from astropy.coordinates import get_sun, SkyCoord
import matplotlib.pyplot as plt
import numpy as np
from typing import Iterable, SupportsFloat as Numeric



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

    #date and time in UTC
    # date = datetime.datetime(2025, 1, 18, 12, 45, 0)
    date:datetime = datetime.fromtimestamp(tstamp,tz = timezone.utc)
    time:Time = Time(date, scale='utc')

    # Create an AltAz frame
    altaz:AltAz = AltAz(obstime=time, location=location)

    # Get the Sun's position in AltAz coordinates
    sun:SkyCoord = get_sun(time)
    sun_altaz:SkyCoord = sun.transform_to(altaz)

    # Extract the zenith angle (90 - altitude)
    # zenith_angle = 90 * u.deg - sun_altaz.alt
    zenith:float = np.abs(sun_altaz.alt.deg - 90) # type: ignore


    # print(f"{time}")
    # lt = date.astimezone(tz = pytz.timezone('Europe/Stockholm'))
    # print(f'{lt} -- {zenith_angle}')
    return zenith


if __name__ == '__main__':
    # Kiruna, Sweden coordinates
    longitude:Numeric = 20.41
    latitude:Numeric = 67.84 
    elevation:Numeric = 420 # Approximate elevation

    test_date:datetime = datetime(2025, 1, 27, tzinfo=timezone.utc) 
    res:Iterable[datetime] = [test_date]
    for i in range(24*4):
        t = res[-1] + timedelta(minutes = 30)
        res.append(t)
    tstamps:Iterable[Numeric] = [datetime.timestamp(r) for r in res]
    sza:Iterable[Numeric] = [solar_zenith_angle(t, latitude,longitude,elevation) for t in tstamps]


    plt.plot(res,sza) # type: ignore
    plt.axhline(90, ls='-.')
    plt.gcf().autofmt_xdate()
    plt.xlabel('time (UTC)')
    plt.ylabel('Solar Zenith Angle (deg)')
    plt.title('Location: Kiruna, Sweden')
    plt.ylim(np.max(sza)+5, np.min(sza)-5)
    

