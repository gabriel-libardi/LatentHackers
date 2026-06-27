# LatentHackers

A workspace for processing and analyzing the Modified Swiss Dwellings (MSD) dataset for floorplan and room layout generation tasks.

## Contents

- [test.ipynb](file:///home/manifoldismay/Documents/Projects/LatentHackers/test.ipynb): A Jupyter notebook demonstrating how to:
  - Load the dwelling geometry dataset using `pandas` and `geopandas`.
  - Parse WKT (Well-Known Text) spatial data.
  - Buffer room geometries to generate solid apartment outlines.
  - Visualize and compare input outlines versus target room partitions using `matplotlib`.
- [mds_V2_5.372k.csv](file:///home/manifoldismay/Documents/Projects/LatentHackers/mds_V2_5.372k.csv): Local copy of the Modified Swiss Dwellings dataset.
- [LICENSE](file:///home/manifoldismay/Documents/Projects/LatentHackers/LICENSE): MIT License.

## Prerequisites

To run the notebook and analyze the dataset, install the required packages:

```bash
pip install pandas geopandas shapely matplotlib kagglehub
```

## Getting Started

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```
2. Open and run [test.ipynb](file:///home/manifoldismay/Documents/Projects/LatentHackers/test.ipynb).
