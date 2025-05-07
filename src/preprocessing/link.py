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
import math
from shapely.geometry import Point as POINT
from shapely.ops import substring
from src.preprocessing.spatial_line import SpatialLine

class Link(SpatialLine):
    """
    Class representing a link in a transportation network.
    Inherits from the SpatialLine class.
    """
    Identification = 0
    def __init__(self, start_point: POINT, end_point: POINT, link_id: int = None):
        """
        Initializes a Link object.

        Args:
            start_point (POINT): The starting point of the link.
            end_point (POINT): The ending point of the link.
        """
        super().__init__(start_point, end_point)
        if link_id is None:
            Link.Identification += 1
            self.link_id = Link.Identification
        else:
            self.link_id = link_id
            Link.Identification = max(Link.Identification, link_id)
        self.cells = []

    def add_cell(self, cell):
        """
        Adds a cell to the link.

        Args:
            cell (Cell): The cell to be added to the link.
        """
        self.cells.append(cell)

    def load_cells_by_length(self, cell_length: float):
        """
        Divides the link into cells based on the specified cell length.

        Args:
            cell_length (float): The length of each cell in meters.
        """
        from src.preprocessing.cell import Cell
        if cell_length <= 0:
            raise ValueError("Cell length must be positive")

        number_of_cells = int(self.length_meters / cell_length)
        distance = self.length_meters
        number_of_cells = max(1, math.ceil(distance / cell_length))
        cells = []
        for i in range(number_of_cells):
            start_dist = i * cell_length
            end_dist = min((i + 1) * cell_length, distance)
            cell_geom = substring(self.line, start_dist, end_dist)
            coords = list(cell_geom.coords)
            coords = list(map(lambda x: self._transformer_to_source.transform(x[0], x[1]), coords))
            cell = Cell(POINT(coords[0]), POINT(coords[-1]))
            cell.set_link(self)
            self.add_cell(cell)
            cells.append(cell)
        return cells

    def load_cells_by_number(self, number_of_cells: int):
        """
        Divides the link into a specified number of cells.

        Args:
            number_of_cells (int): The number of cells to divide the link into.
        """
        from src.preprocessing.cell import Cell
        distance = self.length_meters
        cells = []
        cell_length = distance / number_of_cells
        for i in range(number_of_cells):
            start_dist = i * cell_length
            end_dist = min((i + 1) * cell_length, distance)
            cell_geom = substring(self.line, start_dist, end_dist)
            coords = list(cell_geom.coords)
            coords = list(map(lambda x: self._transformer_to_source.transform(x[0], x[1]), coords))
            cell = Cell(POINT(coords[0]), POINT(coords[-1]))
            cell.set_link(self)
            self.add_cell(cell)
            cells.append(cell)
        return cells

    def __str__(self):
        """
        Returns a string representation of the Link object.
        """
        return (
            f"Link from {self.line.coords[0]} to {self.line.coords[1]} "
            f"with length {self.length_meters} meters"
        )
    def __repr__(self):
        """
        Returns a string representation of the Link object for debugging.
        """
        return (
            f"Link Id: {self.link_id}, "
            f"Length: {self.length_meters} meters"
        )
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

    def __eq__(self, other):
        """
        Compares two Link objects for equality.
        """
        if not isinstance(other, Link):
            return False
        return (
            self.line.equals(other.line) and
            self.length_meters == other.length_meters
        )

    def __hash__(self):
        """
        Returns a hash value for the Link object.
        """
        return super().__hash__()
