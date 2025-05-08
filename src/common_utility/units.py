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
    VEH = u.def_unit("veh", 1)

    # Density
    VEH_PER_KM = VEH / KM
    VEH_PER_M = VEH / M

    # Flow
    VEH_PER_HR = VEH / HR
    VEH_PER_SEC = VEH / S

    # Speed
    KM_PER_HR = KM / HR
    M_PER_SEC = M / S


    # Type:
    Quantity = u.Quantity
