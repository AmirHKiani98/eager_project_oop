"""
FILE: spatial_line.py
DESCRIPTION: This module defines the SpatialLine class, which represents a spatial line
             defined by a list of intersection locations.

CLASSES:
    - SpatialLine: Represents a spatial line defined by a list of intersection locations.

USAGE:
    This module can be used to create and manage spatial lines by providing a list of
    intersection locations. It is useful in applications involving spatial geometry or
    path representation.

"""
from shapely.ops import transform

from shapely.geometry import LineString as LINESTRING
from shapely.geometry import Point as POINT
from pyproj import Transformer

class SpatialLine:

    """
    Represents a spatial line defined by a list of intersection locations.

    Attributes:
        intersection_locations (list): A list of coordinates or points
                            that define the intersections along the spatial line.
    """

    def __init__(self, _from: POINT, _to: POINT, crs="EPSG:4326"):
        """
        Initializes a SpatialLine object.

        Args:
            _from (POINT): A Point object representing the starting point of the spatial line.
            _to (POINT): A Point object representing the ending point of the spatial line.
            crs (str): The coordinate reference system of the points. Default is "EPSG:4326".
        """

        # Define the transformer to convert coordinates to a metric CRS (e.g., EPSG:3857)
        transformer = Transformer.from_crs(crs, "EPSG:3857", always_xy=True)

        # Transform the points to the metric CRS
        _from_metric = transform(transformer.transform, _from)
        _to_metric = transform(transformer.transform, _to)

        # Create the LineString in the metric CRS
        self.intersection_locations = LINESTRING([_from_metric, _to_metric])

        # Calculate the length in meters
        self.length_meters = self.intersection_locations.length

    
