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
from abc import abstractmethod
from shapely.ops import transform
from shapely.geometry import LineString as LINESTRING
from shapely.geometry import Point as POINT
from pyproj import Transformer
from src.common_utility.units import Units


class SpatialLine:

    """
    Represents a spatial line defined by a list of intersection locations.

    Attributes:
        intersection_locations (list): A list of coordinates or points
                            that define the intersections along the spatial line.
    """

    def __init__(self, _from: POINT, _to: POINT):
        """
        Initializes a SpatialLine object.

        Args:
            _from (POINT): A Point object representing the starting point of the spatial line.
            _to (POINT): A Point object representing the ending point of the spatial line.
            crs (str): The coordinate reference system of the points. Default is "EPSG:4326".
        """

        self.source_crs = "EPSG:4326"
        self.metric_crs = "EPSG:3857"
        self._transformer_to_metric = Transformer.from_crs(
            self.source_crs, self.metric_crs, always_xy=True
        )
        self._transformer_to_source = Transformer.from_crs(
            self.metric_crs, self.source_crs, always_xy=True
        )
        # Keep the original points in their original CRS
        self._from = _from
        self._to = _to

        # Transform points to metric CRS for length calculation
        self._from_metric = transform(self._transformer_to_metric.transform, _from)
        self._to_metric = transform(self._transformer_to_metric.transform, _to)

        self.line = LINESTRING([self._from_metric, self._to_metric])
        self.length_meters = self.line.length * Units.M

    def get_distance(self, point: POINT) -> Units.Quantity:
        """
        Calculates the distance from the spatial line to a given point in meters.

        Args:
            point (POINT): A Point object representing the point to calculate the distance to.

        Returns:
            float: The distance from the spatial line to the point in meters.
        """
        # Ensure the point is within valid latitude and longitude ranges
        if point.x > 180 or point.x < -180 or point.y > 90 or point.y < -90:
            raise ValueError("Point must be within valid latitude and longitude ranges.")
        # Ensure the distance is calculated in meters
        point_metric = transform(self._transformer_to_metric.transform, point)
        return self.line.distance(point_metric) * Units.M

    @abstractmethod
    def __eq__(self, other):
        pass

    def __hash__(self):
        """
        Returns a hash value for the SpatialLine object.
        """
        return hash((self._from, self._to, self.length_meters))


    def get_from(self) -> POINT:
        """
        Returns the starting point of the cell.

        Returns:
            POINT: The starting point of the cell.
        """
        return self._from

    def get_to(self) -> POINT:
        """
        Returns the ending point of the cell.
        Returns:
            POINT: The ending point of the cell.
        """
        return self._to

    def get_line_source(self) -> LINESTRING:
        """
        Returns the line in its original CRS.
        Returns:
            LINESTRING: The line in its original CRS.
        """
        return transform(self._transformer_to_source.transform, self.line)

    def get_distance_from_beginning(self, point: POINT):
        """
        Returns the distance from the beginning of the link to the specified cell.
        """
        if point.x > 180 or point.x < -180 or point.y > 90 or point.y < -90:
            raise ValueError("Point must be within valid latitude and longitude ranges.")
        # Ensure the distance is calculated in meters
        point = transform(self._transformer_to_metric.transform, point)
        return self._from_metric.distance(point) * Units.M
