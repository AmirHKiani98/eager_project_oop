"""
This module defines custom units for vehicle measurements.
It uses the Astropy library to define units for vehicles,
vehicles per kilometer, and vehicles per hour.
This module is part of a larger traffic simulation framework
and is intended to be used in conjunction with other modules
and classes within the framework.
"""
from astropy import units as u



class Units:
    """
    This class defines custom units for vehicle measurements
    using the Astropy library. It includes units for vehicles,
    vehicles per kilometer, vehicles per hour, and other
    relevant measurements. The units are defined in a way that
    allows for easy conversion and manipulation within the
    traffic simulation framework.
    """
    # Base units
    KM = u.kilometer
    M = u.meter
    HR = u.hour
    S = u.second

    # Density
    PER_KM = 1 / KM
    PER_M = 1 / M

    # Flow
    PER_HR = 1 / HR
    PER_SEC = 1 / S

    # Speed
    KM_PER_HR = KM / HR
    M_PER_SEC = M / S


    # Type:
    Quantity = u.Quantity
