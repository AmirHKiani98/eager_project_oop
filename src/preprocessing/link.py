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
import logging
from typing import Optional
from shapely.geometry import Point as POINT
from shapely.ops import substring
from rich.logging import RichHandler
from src.preprocessing.spatial_line import SpatialLine
from src.common_utility.units import Units

logging.basicConfig(
    level="DEBUG",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("rich")

class Link(SpatialLine):
    """
    Class representing a link in a transportation network.
    Inherits from the SpatialLine class.
    """
    Identification = 0
    def __init__(
        self,
        start_point: POINT,
        end_point: POINT,
        link_id: Optional[int] = None,
        tl: bool = True
    ):
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
        self.cells = {}
        self.tl = tl

    def add_cell(self, cell):
        """
        Adds a cell to the link.

        Args:
            cell (Cell): The cell to be added to the link.
        """
        self.cells[cell.cell_id] = cell

    def is_tl(self):
        """
        Returns whether the link has a traffic light.

        Returns:
            bool: True if the link has a traffic light, False otherwise.
        """
        return self.tl

    def load_cells_by_length(self, cell_length: Units.Quantity):
        """
        Divides the link into cells based on the specified cell length.

        Args:
            cell_length (Units.Quantity): The length of each cell in meters.
        """
        from src.preprocessing.cell import Cell
        if not isinstance(cell_length, Units.Quantity):
            raise TypeError("Cell length must be a Units.Quantity")
        length_meters_value = self.length_meters.to(Units.M).value
        cell_length_value = cell_length.to(Units.M).value
        number_of_cells = length_meters_value / cell_length_value
        distance = length_meters_value
        number_of_cells = max(1, math.ceil(distance / cell_length_value))
        cells = []
        for i in range(number_of_cells):
            start_dist = i * cell_length_value
            end_dist = min((i + 1) * cell_length_value, distance)
            cell_geom = substring(self.line, start_dist, end_dist)
            coords = list(cell_geom.coords)
            coords = list(map(lambda x: self._transformer_to_source.transform(x[0], x[1]), coords))
            cell = Cell(POINT(coords[0]), POINT(coords[-1]), cell_id=len(self.cells)+1)
            cell.set_link(self)
            self.add_cell(cell)
            cells.append(cell)
        self.average_cell_length = sum([cell.get_length().to(Units.M).value for cell in self.cells])/len(self.cells)
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
        self.average_cell_length = sum([cell.get_length().to(Units.M).value for cell in self.cells])/len(self.cells)
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

    def get_cell_length(self, cell_id):
        """
        Returns the length of the specified cell in the link.
        """
        return self.cells[cell_id].length_meters

    def get_cell(self, cell_id):
        """
        Returns the specified cell in the link.
        """
        return self.cells[cell_id]
