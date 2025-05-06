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