# Geospatial Preprocessing Module

This module provides a robust framework for handling geospatial data within transportation networks. It enables the user to load intersection points, build links between them, segment links into spatial cells, and cache the results for efficient reuse. It is primarily designed to support traffic simulation and modeling applications that require a spatially structured network.

---

## 📦 Module Overview

### Key Components

- **DataLoader**: Main entry point for downloading, loading and processing the data from pNEUMA.
- **GeoLoader**: Main entry point for loading, processing, and saving geospatial data.
- **Link**: Represents a directed edge between two intersection points, optionally tagged with traffic light presence.
- **Cell**: Subdivision of a link used in traffic simulation models (e.g., CTM, LTM, PQ).
- **SpatialLine**: Abstract base class for any geospatial linear feature, including utility methods for transformations and measurements.
- **fill_missing_timestamps**: Utility to generate uniformly spaced time-series data with placeholder values.


The DataLoader class handles end-to-end traffic data preprocessing from raw downloads to structured simulation inputs. It supports five traffic simulation models: CTM, Point Queue (PQ), Spatial Queue (SQ), Link Transmission Model (LTM), and Payne-Whitham (PW).
Features:
* Download and cache raw pNEUMA vehicle trajectory files from EPFL server
* Explode compressed trajectory columns into flat Polars DataFrames
* Clean and interpolate missing timestamps per vehicle
* Assign each coordinate to its closest link and cell using GeoLoader
* Filter vehicles that are not aligned with the main corridor
* Compute occupancy, entry/exit, and density statistics per cell and timestamp
* Detect traffic light status from average speed thresholds
* Aggregate cumulative vehicle counts for queue-based models
* Prepare task lists for each traffic simulation model (CTM, PQ, SQ, LTM, PW)

| Filename Suffix                              | Description                                 |
| -------------------------------------------- | ------------------------------------------- |
| `_exploded.csv`                              | Flattened trajectory with timestamped rows  |
| `_withlinkcell_<hash>.csv`                   | Data joined with nearest link and cell IDs  |
| `_vehicle_on_corridor_<hash>.csv`            | Filtered vehicles moving along the corridor |
| `_density_entry_exit_<hash>.csv`             | Occupancy, entry/exit, density per cell     |
| `_cumulative_counts_<params>_<hash>.csv`     | Link-level cumulative entry/exit counts     |
| `_traffic_light_status_<hash>.csv`           | Avg. speed per traffic-light per time       |
| `_processed_traffic_light_status_<hash>.csv` | Binary green/red status for traffic lights  |
| `_prepared_*_tasks_*.json`                   | Final task files per traffic model          |

1. Initialize GeoLoader
```py
geo = GeoLoader(locations=[...], cell_length=25)
```

2. Initialize DataLoader
```py
from src.model.params import Parameters
from src.preprocessing.data_loader import DataLoader

params = Parameters()
dl = DataLoader(
    fp_location="d1",
    fp_date="20181029",
    fp_time="0800_0830",
    geo_loader=geo,
    params=params
)
```

3. Prepare Simulation Tasks
```py
dl.prepare("CTM", "d1", "20181029", "0800_0830")
tasks = dl.tasks
```

