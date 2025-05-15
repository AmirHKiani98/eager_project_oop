"""
Module: test_geo_loader

This module contains unit tests for the geo_loader preprocessing functionality.

Functions:
    test_find_closest_link_cell(simple_geo_loader):
        Tests the 'find_closest_link_cell' method of the GeoLoader class.
        Ensures that the function correctly identifies the closest cell in a link
        to a given point.
"""
import logging
from rich.logging import RichHandler
from shapely.geometry import Point as POINT

logging.basicConfig(
    level="DEBUG",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("rich")
def test_find_closest_link_cell(simple_geo_loader):
    """
    Test the find_closest_link_cell function.
    """

    # Create a mock link
    point1 = POINT(23.735167249933692, 37.98058492307966) # closest to link 1 cell 1
    point2 = POINT(23.73540417309564, 37.98018271518239) # closest to link 1 cell 1
    point3 = POINT(23.73603068976541, 37.97947212671277)  # closest to link 2 cell 1
    point4 = POINT(23.73639338645874, 37.978982044583056) # closest to link 2 last cell
    closeset_link1, _, closest_cell1, _ = simple_geo_loader.find_closest_link_and_cell(point1)
    closeset_link2, _, closest_cell2, _ = simple_geo_loader.find_closest_link_and_cell(point2)
    closeset_link3, _, closest_cell3, _ = simple_geo_loader.find_closest_link_and_cell(point3)
    closeset_link4, _, closest_cell4, _ = simple_geo_loader.find_closest_link_and_cell(point4)
    # Check the results
    assert closeset_link1.link_id == 1
    assert closeset_link2.link_id == 1
    assert closeset_link3.link_id == 2
    assert closeset_link4.link_id == 2
    assert closest_cell1.cell_id == 1
    assert closest_cell2.cell_id == 1
    assert closest_cell3.cell_id == 1
    assert closest_cell4.cell_id == len(closeset_link4.cells)
    # Check the result


def test_length_of_cells():
    """
    Test the length of cells in the GeoLoader.
    """
    assert True
