"""
Handles loading and preprocessing of geospatial data.
This module provides functionality to load geospatial data from various sources,
transform coordinate reference systems, and perform spatial operations.

"""
import polars as pl
import matplotlib.pyplot as plt
from shapely.geometry import Point as POINT
from src.preprocessing.cell import Cell
from src.preprocessing.link import Link
from src.preprocessing.data_loader import DataLoader
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
                fp_location,
                fp_date,
                fp_time,
                intersection_locations: list[POINT] = None,
                cell_length: float = None,
                number_of_cells: int = None):
        self.data_loader = DataLoader(fp_location, fp_date, fp_time)
        self.intersection_locations = intersection_locations
        self.links = []
        self.cells = []
        self.cell_length = cell_length
        self.number_of_cells = number_of_cells
        self._load_links()
        self._load_cells()

    def _load_links(self):
        """
        Load links from the data loader.
        """
        for index in range(len(self.intersection_locations) - 1):
            start_point = self.intersection_locations[index]
            end_point = self.intersection_locations[index + 1]
            link = Link(start_point, end_point)
            self.links.append(link)
    

    def _load_cells_by_length(self):
        for link in self.links:
            distance = link.length_meters
            number_of_cells = int(distance / self.cell_length)
            for i in range(number_of_cells):
                cell_start = link.line.interpolate(i * self.cell_length)
                cell_end = link.line.interpolate(min((i + 1) * self.cell_length, distance))
                cell = Cell(cell_start, cell_end)
                cell.set_link(link)
                link.add_cell(cell)
                self.cells.append(cell)


    def _load_cells_by_number(self):
        """
        Placeholder method to divide each link into a specified number 
        of cells.
        """
        for link in self.links:
            distance = link.length_meters
            cell_length = distance / self.number_of_cells
            for i in range(self.number_of_cells):
                cell_start = link.line.interpolate(i * cell_length)
                cell_end = link.line.interpolate(min((i + 1) * cell_length, distance))
                cell = Cell(cell_start, cell_end)
                cell.set_link(link)
                link.add_cell(cell)
                self.cells.append(cell)
    
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
        for link in self.links:
            x, y = link.line.xy
            ax.plot(x, y, color='blue', linewidth=2, alpha=0.5)
        for cell in self.cells:
            x, y = cell.line.xy
            ax.plot(x, y, color='red', linewidth=3, alpha=0.2)
        plt.show()
        plt.axis('equal')

    def save(self):
        """
        Saves the links and cells to a file.
        """
        # Placeholder for saving logic
        pass


if __name__ == "__main__":
    # Example usage
    intersection_locations = pl.read_csv(".cache/traffic_lights.csv").to_numpy().tolist()
    intersection_locations = [POINT(loc[1], loc[0]) for loc in intersection_locations]
    geo_loader = GeoLoader(
        fp_location="d1",
        fp_date="20181029",
        fp_time="0800_0830",
        intersection_locations=intersection_locations,
        cell_length=20
    )
    links = geo_loader.get_links()
    cells = geo_loader.get_cells()
    print(f"Loaded {len(links)} links and {len(cells)} cells.")
    geo_loader.draw()