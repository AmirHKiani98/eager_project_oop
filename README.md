# Eager Project OOP
This repository implements an object-oriented approach to data processing and analysis. The codebase is modularized for clarity and maintainability, with each major functionality separated into its own module. The main modules include preprocessing, common utilities, modeling, and visualization. Each module is documented with its own README file, providing detailed usage instructions and examples.

The project is designed for extensibility, allowing users to easily add new features or modify existing ones. To get started, review the documentation in each module and follow the provided examples for integrating the modules into your workflow.

This project is organized into several modules, each with its own README file for detailed documentation. Below are the links to the respective README files:

- [Preprocessing Module](./src/preprocessing/README.md)  
- [Common Utility Module](./src/common_utility/README.md)  
- [Model Module](./src/model/README.md)  
- [Visualization Module](./src/visualization/README.md)  

Refer to each module's README for specific details and usage instructions.  

The system uses drone-collected vehicle trajectory data from the [EPFL pNEUMA dataset](https://open-traffic.epfl.ch) and supports efficient preprocessing, parameter calibration, and multiprocessing-based simulation runs.

---

## 📁 Entry Point: `src/main.py`

This script is the primary interface for users. It handles:

- Argument parsing
- Loading traffic light geolocation data
- Initializing the data pipeline (`GeoLoader` + `DataLoader`)
- Choosing and executing the appropriate traffic model
- Running simulations and calibrations with multiprocessing

---

## 🚀 Quick Start

### ✅ Install Dependencies

```bash
pip install -r requirements.txt
```

Make sure `polars`, `shapely`, and `tqdm` are installed. Also ensure your dataset is pre-downloaded or reachable via the DataLoader.

## Example Run
### Run the Cell Transmission Model:
```bash
python -m src.main \
  --model ctm \
  --fp-location d1 \
  --fp-date 20181029 \
  --fp-time 0800_0830 \
  --fp-geo .cache/traffic_lights.csv
```
### Run Payne-Whitham model with calibration enabled:
```bash
python -m src.main \
  --model pw \
  --calibration
```

### Run all models in sequence:
```bash
python -m src.main \
  --model ctm,pq,sq,ltm,pw
```

| Argument        | Type   | Default                     | Description                                                    |
| --------------- | ------ | --------------------------- | -------------------------------------------------------------- |
| `--model`       | `str`  | `ctm`                       | One of: `ctm`, `pq`, `sq`, `ltm`, `pw` or comma-separated list |
| `--batch-size`  | `int`  | `50000`                     | Batch size for multiprocessing                                 |
| `--fp-location` | `str`  | `d1`                        | Dataset location (EPFL folder name)                            |
| `--fp-date`     | `str`  | `20181029`                  | Date string in `YYYYMMDD` format                               |
| `--fp-time`     | `str`  | `0800_0830`                 | Time interval string                                           |
| `--fp-geo`      | `str`  | `.cache/traffic_lights.csv` | Path to traffic light geolocation CSV                          |
| `--calibration` | `flag` | `False`                     | If set, enables parameter calibration for the model            |
