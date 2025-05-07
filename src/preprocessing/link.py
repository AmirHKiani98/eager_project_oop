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
    
    def __str__(self):
        """
        Returns a string representation of the Link object.
        """
        return f"Link from {self.line.coords[0]} to {self.line.coords[1]} with length {self.length_meters} meters"
    
    def __repr__(self):
        """
        Returns a string representation of the Link object for debugging.
        """
        return f"Link(start_point={self.line.coords[0]}, end_point={self.line.coords[1]}, length_meters={self.length_meters})"
    
    def __len__(self):
        """
        Returns the number of cells in the link.
        """
        return len(self.cells)
    
    def __getitem__(self, index):
        """
        Returns the cell at the specified index.
        """
        return self.cells[index]
