# 📊 Plotter Module for Traffic Model Visualization

The `Plotter` class provides a comprehensive framework for **visualizing traffic simulation results**, especially those derived from macroscopic models like Point Queue, Spatial Queue, CTM, LTM, and PW.

It supports:
- Generating per-model RMSE/error heatmaps
- Saving calibration errors
- Animating vehicle movements from raw trajectory data
- Persisting all model errors in a shared JSON file (`all_errors.json`)

---

## 📁 File Location
```bash
src/visualization/plotter.py
```

## Features
✅ Model-Specific Heatmap Generation
* PointQueue and SpatialQueue: error between sending/receiving flow and occupancy
* CTM: per-cell squared density error
* LTM: flow and density error derived from Lax-Hopf discretization
* PW: squared error between predicted and next densities

✅ Centralized Error Logging
* Automatically logs average RMSE per calibration run
* Saves to .cache/all_errors.json for reuse or plotting

✅ Vehicle Animation
* Plots a vehicle's movement over time based on lon/lat and trajectory data
* Saves animation as animation.gif

## Dependencies
This module is mainly dependent on `matploltib`, `searborn`, `polars`, `pandas`, `shapely` and `tqdm`.

## Example Usage
```py
from src.visualization.plotter import Plotter
from src.preprocessing.geo_loader import GeoLoader
from shapely.geometry import Point

geo_loader = GeoLoader(
    locations=[Point(...), Point(...)],  # intersection coordinates
    cell_length=40.0
)
plotter = Plotter(cache_dir=".cache", geo_loader=geo_loader)

plotter.plot_error_ltm(
    data_file_name="d1_20181029_0800_0830",
    hash_parmas="dcca17e9025816395dbe6a5a465c2450",
    hash_geo="682a48de",
    traffic_model="LTM"
)
```
## Output structure
.cache/
├── all_errors.json
├── results/
│   └── {data_file_name}/
│       └── {ModelName}/
│           ├── error.png
│           └── Link_{id}.png
