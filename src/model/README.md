# Traffic Models (`src/model/`)

This module implements various **macroscopic traffic flow models** for simulating and evaluating traffic dynamics on road networks using the pNEUMA dataset. Each model inherits from a shared abstract base class, `TrafficModel`, which defines the structure for initialization, flow computation, and simulation orchestration.

## 📁 Module Structure

src/model/
├── traffic_model.py # Abstract base class defining the model interface
├── point_queue.py # Point Queue (PQ) model
├── spatial_queue.py # Spatial Queue (SQ) model
├── ctm.py # Cell Transmission Model (CTM)
├── ltm.py # Link Transmission Model (LTM)
├── pw.py # Payne-Whitham second-order model
├── params.py # Configuration and parameter class for simulations


---

## 📌 Core Concepts

### `TrafficModel` (Abstract Base Class)

This is the base class that all models must inherit from. It provides:

- Initialization of simulation context (location, time, data loader, plotter)
- Utility methods for computing outflow and retrieving geo-information
- Multiprocessing-based simulation engine
- Calibration workflow across parameter grids
- Caching and result serialization
- Abstract methods: `predict`, `compute_flow`, `run`

---

## Implemented Models

### 1. **Point Queue (PQ)**
- Simplified model with single density and flow per link per timestep.
- Ignores spatial variation.
- Fastest to compute, ideal for large-scale networks.

### 2. **Spatial Queue (SQ)**
- Builds upon PQ with receiving flow constraints.
- Accounts for downstream capacity limitations and jam densities.
- Suitable for corridor-level congestion studies.

### 3. **Cell Transmission Model (CTM)**
- Discretizes links into cells.
- Captures shockwave propagation and spatial queue buildup.
- Good trade-off between resolution and performance.

### 4. **Link Transmission Model (LTM)**
- Works at link granularity but applies Lax-Hopf-based logic for flow.
- Incorporates shockwave behavior more accurately than PQ/SQ with fewer cells than CTM.

### 5. **Payne-Whitham (PW)**
- A second-order macroscopic model that accounts for speed, density, and acceleration.
- More sensitive to parameter tuning.
- Useful for capturing advanced driver behavior and velocity waves.

---

## Calibration & Evaluation

The base class includes a method:
```python
model.run_calibration()
```
which performs a grid search across combinations of:
* Free-flow speed (u_f)
* Jam density (k_j)
* Wave speed (w)
* Maximum flow rate (q_max)

All calibrated results are cached and visualized using the Plotter class. Mean Absolute Error (MAE) is used as the default evaluation metric.

## Parameter Management

The Parameters class in params.py:
* Stores and validates simulation parameters
* Computes derived quantities such as:
* Time step duration
* Flow capacity
* Jam density per cell
* Automatically saves metadata to .cache/params/ as JSON using a hash of parameter values.

## Running a Model
```py
from src.model.spatial_queue import SpatialQueue
from src.preprocessing.data_loader import DataLoader

dl = DataLoader(...)
model = SpatialQueue(dl, fp_location="region1", fp_date="2024-10-24", fp_time="08:00")
model.run_with_multiprocessing()
```
## Outputs
Each model outputs:
* Link-level or cell-level flow/density/inflow/outflow
* JSON result files stored in:
```bash
.cache/{ModelName}/{location_date_time}_{geo_hash}_{param_hash}.json
```
* Plots for density error and model behavior using `Plotter` class.