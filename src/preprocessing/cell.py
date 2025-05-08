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
from typing import Optional
from shapely.geometry import Point as POINT
from src.preprocessing.spatial_line import SpatialLine
from src.preprocessing.link import Link
class Cell(SpatialLine):
    """
    Class representing a cell in a transportation network.
    Inherits from the SpatialLine class.
    """
    Identification = 0
    def __init__(self, start_point: POINT, end_point: POINT, cell_id: Optional[int] = None):
        """
        Initializes a Cell object.

        Args:
            start_point (POINT): The starting point of the cell.
            end_point (POINT): The ending point of the cell.
            cell_id (Optional[int]): The identifier for the cell.
        """
        super().__init__(start_point, end_point)
        self.link = None
        self.cell_id = cell_id

    def set_link(self, link: Link):
        """
        Sets a link to the cell.

        Args:
            link (Link): The link to be added to the cell.
        """
        if not isinstance(link, Link):
            raise TypeError("link must be an instance of Link")
        self.link = link


    def __str__(self):
        """
        Returns a string representation of the Cell object.

        Returns:
            str: String representation of the Cell object.
        """
        return (
            f"Cell(start_point={self._from}, end_point={self._to}, "
            f"length_meters={self.length_meters})"
        )

    def __eq__(self, other):
        if not isinstance(other, Cell):
            return False
        return (
            self._from == other._from and
            self._to == other._to and
            self.length_meters == other.length_meters
        )
