"""
Module: link
This module defines the Link class, which represents a link in a transportation network.
The Link class inherits from the SpatialLine class and provides functionality for working
with spatial links defined by a start and end point.
Classes:
    - Link: Represents a transportation network link, inheriting from SpatialLine.
Dependencies:
    - src.preprocessing.spatial_line.SpatialLine: Base class for spatial line representation.
    - shapely.geometry.Point: Used to define the start and end points of the link.
"""
from shapely.geometry import Point as POINT

from src.preprocessing.spatial_line import SpatialLine

class Link(SpatialLine):
    """
    Class representing a link in a transportation network.
    Inherits from the SpatialLine class.
    """
    def __init__(self, start_point: POINT, end_point: POINT):
        """
        Initializes a Link object.

        Args:
            start_point (POINT): The starting point of the link.
            end_point (POINT): The ending point of the link.
        """
        super().__init__(start_point, end_point)
        self.cells = []

    def add_cell(self, cell):
        """
        Adds a cell to the link.

        Args:
            cell (Cell): The cell to be added to the link.
        """
        self.cells.append(cell)
