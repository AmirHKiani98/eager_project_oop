# Author: Amirhossein Kiani/kiani014@umn.edu
// Data: May05th, 2024
# Preprocessing Module

This module provides utilities and classes for preprocessing traffic data, handling spatial relationships, and preparing data for traffic flow simulation and analysis.

## File Descriptions

- **`cell.py`**: Defines the `Cell` class representing a road segment unit in the traffic network, including geometry and length-related methods.

- **`data_loader.py`**: Contains the `DataLoader` class which manages the entire traffic data preparation pipeline, including downloading raw data, exploding vehicle trajectories, mapping points to spatial features, and computing per-cell statistics like density, entry, and exit counts.

- **`geo_loader.py`**: Defines the `GeoLoader` class for managing spatial data. It includes methods for spatial indexing, assigning links and cells to geographic coordinates, and computing proximity metrics.

- **`intersection.py`**: Defines the `Intersection` class to represent intersections in the network and compute traffic light influence using geometric rules.

- **`link.py`**: Implements the `Link` class that models road links consisting of multiple cells. Includes methods to calculate distances, manage constituent cells, and extract geometric properties.

- **`spatial_line.py`**: Contains geometric helper classes like `SpatialLine` used to represent linear road geometries and compute midpoints and distances.

- **`utility.py`**: Utility functions such as timestamp normalization, time-based grouping, and trajectory data interpolation for preprocessing.