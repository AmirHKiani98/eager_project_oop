"""
This module defines the `Cell` class, which represents a cell in a transportation network.
The `Cell` class inherits from the `SpatialLine` class and provides functionality for
initializing a cell with a starting and ending point.

Classes:
    - Cell: Represents a cell in a transportation network.

Dependencies:
    - shapely.geometry.Point: Used to define the starting and ending points of the cell.
    - src.preprocessing.spatial_line.SpatialLine: Base class for the `Cell` class.
"""
from shapely.geometry import Point as POINT
from src.preprocessing.spatial_line import SpatialLine
from src.preprocessing.link import Link
class Cell(SpatialLine):
    """
    Class representing a cell in a transportation network.
    Inherits from the SpatialLine class.
    """
    def __init__(self, start_point: POINT, end_point: POINT):
        """
        Initializes a Cell object.

        Args:
            start_point (POINT): The starting point of the cell.
            end_point (POINT): The ending point of the cell.
        """
        super().__init__(start_point, end_point)
        self.link = None

    def set_link(self, link: Link):
        """
        Sets a link to the cell.

        Args:
            link (Link): The link to be added to the cell.
        """
        if not isinstance(link, Link):
            raise TypeError("link must be an instance of Link")
        self.link = link

    def get_distance(self, point: POINT) -> float:
        """
        Calculates the distance from the cell to a given point in meters.

        Args:
            point (POINT): The point to calculate the distance to.

        Returns:
            float: The distance from the cell to the point in meters.
        """
        # Ensure the distance is calculated in meters
        return self.line.distance(point) * 1  # Assuming the CRS is in meters

    def __str__(self):
        """
        Returns a string representation of the Cell object.

        Returns:
            str: String representation of the Cell object.
        """
        return f"Cell(start_point={self._from}, end_point={self._to}, length_meters={self.length_meters})"
