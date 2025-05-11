"""
Handles loading and preprocessing of geospatial data.
This module provides functionality to load geospatial data from various sources,
transform coordinate reference systems, and perform spatial operations.

"""
import os
import random
import hashlib
import logging
from typing import Optional
import polars as pl
import matplotlib.pyplot as plt
from shapely.geometry import Point as POINT
from rich.logging import RichHandler
from src.preprocessing.cell import Cell
from src.preprocessing.link import Link
from src.common_utility.units import Units

logging.basicConfig(
    level="DEBUG",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("rich")
class GeoLoader:
    """
    GeoLoader is a class designed to preprocess and load geographical data, 
    including links and cells, based on provided intersection locations and 
    configuration parameters.

    Attributes:
        data_loader (DataLoader): An instance of DataLoader to load location, 
            date, and time data.
        intersection_locations (list[POINT]): A list of geographical points 
            representing intersections.
        links (list[Link]): A list of links created between intersection points.
        cells (list[Cell]): A list of cells created along the links.
        cell_length (float): The length of each cell in meters (used for cell 
            generation by length).
        number_of_cells (int): The number of cells to divide each link into 
            (used for cell generation by number).

    Methods:
        __init__(fp_location, fp_date, fp_time, intersection_locations, 
                 cell_length, number_of_cells):
            Initializes the GeoLoader with file paths and optional 
            configuration parameters.
        _load_links():
            Loads links by creating connections between consecutive 
            intersection locations.
        _load_cells_by_length():
            Divides each link into cells based on the specified cell length.
        _load_cells_by_number():
            Placeholder method to divide each link into a specified number 
            of cells.
    """


    def __init__(self,
                locations: Optional[list[POINT]] = None,
                cell_length: Optional[float] = None,
                number_of_cells: Optional[int] = None,
                testing: bool = False):
        self.locations = locations
        # Check if the link is already saved:
        self.links = {}
        self.links_to_location = {}
        self.cells = []
        self.cell_length = cell_length * Units.M
        self.number_of_cells = number_of_cells
        if self._geo_data_already_exists():
            self._load()
        else:
            self._load_links()
            if self.cell_length is None and self.number_of_cells is None:
                print("Warining: No cell length or number of cells provided.")
            else:
                self._load_cells()
            if not testing:    
                self._save()

    def _load_links(self):
        """
        Load links from the data loader.
        """
        if self.locations is None:
            raise ValueError("No locations provided for loading links.")
        for index in range(len(self.locations) - 1):
            start_point = self.locations[index]
            end_point = self.locations[index + 1]
            link = Link(start_point, end_point, link_id=len(self.links) + 1)
            self.links[link.link_id] = link

    def _load_cells_by_length(self):
        for _, link in self.links.items():
            link_cells = link.load_cells_by_length(self.cell_length)
            self.cells.extend(link_cells)

    def _load_cells_by_number(self):
        """
        Placeholder method to divide each link into a specified number 
        of cells.
        """
        for _, link in self.links.items():
            link_cells = link.load_cells_by_number(self.number_of_cells)
            self.cells.extend(link_cells)

    def _load_cells(self):
        if self.cell_length is not None:
            self._load_cells_by_length()
        elif self.number_of_cells is not None:
            self._load_cells_by_number()
        else:
            raise ValueError("Either one of cell_length or number_of_cells must be provided.")

    def get_links(self):
        """
        Returns the list of links created from the intersection locations.
        """
        return self.links

    def get_cells(self):
        """
        Returns the list of cells created along the links.
        """
        return self.cells

    def draw(self):
        """
        Draws the links and cells on a map.
        """
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 10)
        ax.set_title("Links and Cells")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        for _, link in self.links.items():
            x, y = link.line.xy
            # ax.plot(x, y, color='blue', linewidth=2, alpha=0.5)
        for cell in self.cells:
            x, y = cell.line.xy
            print(f"Link: {cell.link.length_meters} meters, Cell: {cell.length_meters} meters")
            random_color = (random.random(), random.random(), random.random())
            ax.plot(x, y, color=random_color, linewidth=3, alpha=1)
        plt.show()
        plt.axis('equal')

    def _get_hash_str(self):
        # Generate a unique identifier based on intersection locations
        if self.locations is None:
            raise ValueError("No locations provided for generating hash.")
        hash_input = str([(point.x, point.y) for point in self.locations])
        if self.cell_length is not None:
            hash_input += f"_cell_length_{self.cell_length}"
        elif self.number_of_cells is not None:
            hash_input += f"_number_of_cells_{self.number_of_cells}"
        hash_str = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return hash_str

    def get_hash_str(self):
        """
        Returns a unique identifier based on the intersection locations and 
        configuration parameters.
        """
        return self._get_hash_str()

    def _save(self):
        """
        Saves the links and cells to CSV files, including metadata about the configuration.

        The files will include information about the intersection locations, cell length, 
        and number of cells used for processing. File names will include a hash of the 
        intersection locations to avoid overlap.
        """
        hash_str = self._get_hash_str()

        # Save links to a CSV file
        links_data = [
            {"link_id": link.link_id,
            "start_lon": link.get_from().x, "start_lat": link.get_from().y,
             "end_lon": link.get_to().x, "end_lat": link.get_to().y, 
             "length_meters": link.length_meters.to(Units.M).value,}
            for link in self.links.values()
        ]
        links_df = pl.DataFrame(links_data)
        links_df.write_csv(f".cache/links_{hash_str}.csv")

        # Save cells to a CSV file
        cells_data = [
            {
            "link_id": cell.link.link_id,
            "cell_id": cell.cell_id,
            "cell_start_lon": cell.get_line_source().coords[0][0],
            "cell_start_lat": cell.get_line_source().coords[0][1],
            "cell_end_lon": cell.get_line_source().coords[1][0],
            "cell_end_lat": cell.get_line_source().coords[1][1],
            "cell_length_meters": cell.length_meters.to(Units.M).value,
            }
            for cell in self.cells
        ]
        cells_df = pl.DataFrame(cells_data)
        cells_df.write_csv(f".cache/cells_{hash_str}.csv")

        # Save metadata to a separate file
        if self.locations is None:
            raise ValueError("No locations provided for saving metadata.")
        metadata = {
            "locations_count": len(self.locations),
            "cell_length": self.cell_length.to(Units.M).value,
            "number_of_cells": self.number_of_cells,
        }
        metadata_df = pl.DataFrame([metadata])
        metadata_df.write_csv(f".cache/metadata_{hash_str}.csv")

    def _load(self):
        """
        Loads links and cells from saved CSV files.
        """
        hash_str = self._get_hash_str()
        links_df = pl.read_csv(f".cache/links_{hash_str}.csv")

        for row in links_df.iter_rows(named=True):
            start_point = POINT(row['start_lon'], row['start_lat'])
            end_point = POINT(row['end_lon'], row['end_lat'])
            link_id = row['link_id']
            link = Link(start_point, end_point, link_id=len(self.links) + 1)
            self.links[link_id] = link


        cells_df = pl.read_csv(f".cache/cells_{hash_str}.csv")
        for row in cells_df.iter_rows(named=True):
            cell_start = POINT(row['cell_start_lon'], row['cell_start_lat'])
            cell_end = POINT(row['cell_end_lon'], row['cell_end_lat'])
            cell_id = row['cell_id']
            link_id = row['link_id']
            link_found = False
            for _, link in self.links.items():
                if link.link_id == link_id:
                    link_found = True
                    break
            if link_found:
                cell = Cell(cell_start, cell_end, cell_id=cell_id)
                cell.set_link(link)
                link.add_cell(cell)
                self.cells.append(cell)
            else:
                print(f"Link: {link_id} not found for cell {cell_id}")

    def _geo_data_already_exists(self):
        """
        Check if the geospatial data already exists in the cache.
        """
        hash_str = self._get_hash_str()
        if os.path.exists(f".cache/links_{hash_str}.csv") and \
           os.path.exists(f".cache/cells_{hash_str}.csv"):
            return True
        return False

    def find_closest_link_and_cell(
        self, point: POINT
    ) -> tuple[Optional[Link], Units.Quantity, Optional[Cell], Units.Quantity]:
        """
        Finds the closest link to a given point.

        Args:
            point (POINT): The point to find the closest link to.

        Returns:
            tuple[Optional[Link], Units.Quantity, Optional[Cell], Units.Quantity]: 
            The closest link and its distance, and the closest cell 
            and its distance.
        """
        min_distance_link = float("inf")
        closest_link = None
        for _, link in self.links.items():
            if not isinstance(link, Link):
                raise TypeError("Link must be of type Link")
            distance = link.get_distance(point)
            if distance < min_distance_link:
                min_distance_link = distance
                closest_link = link

        if closest_link is None:
            raise ValueError("No link found for the given point.")
        min_distance_cell = float("inf")
        closest_cell = None

        for _, cell in closest_link.cells.items():
            if not isinstance(cell, Cell):
                raise TypeError("Cell must be of type Cell")
            distance = cell.get_distance(point)
            if distance < min_distance_cell:
                min_distance_cell = distance
                closest_cell = cell
        if not isinstance(min_distance_cell, Units.Quantity):
            raise TypeError("Distance must be of type Units.Quantity")
        if not isinstance(min_distance_link, Units.Quantity):
            raise TypeError("Distance must be of type Units.Quantity")
        return (closest_link, min_distance_link, closest_cell, min_distance_cell)

    def get_cell_length(self, cell_id, link_id):
        """
        Get the length of a specific cell.

        Args:
            cell_id (int): The ID of the cell.
            link_id (int): The ID of the link.

        Returns:
            float: Length of the specified cell.
        """
        return self.links[link_id].get_cell_length(cell_id)
    
    def get_link_length(self, link_id):
        """
        Get the length of a specific link.

        Args:
            link_id (int): The ID of the link.

        Returns:
            float: Length of the specified link.
        """
        return self.links[link_id].get_length()

    def is_tl(self, link_id):
        """
        In the future, it will check if the link has a traffic light.
        """
        return self.links[link_id].is_tl()

    def find_closest_location(self, point: POINT) -> tuple[Optional[POINT], float]:
        """
        Finds the closest location to a given point.

        Args:
            point (POINT): The point to find the closest location to.

        Returns:
            tuple[POINT, float]: The closest location and its distance from the point.
        """
        min_distance = float("inf")
        closest_location = None
        for _, link in self.links.items():
            distance = link.get_distance_from_beginning(point)
            if distance < min_distance:
                min_distance = distance
                closest_location = link
        return (closest_location, min_distance)
