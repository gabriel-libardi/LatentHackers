# LatentHackers

A workspace for processing and analyzing the Modified Swiss Dwellings (MSD) dataset for floorplan and room layout generation tasks.

## Contents

- [test.ipynb](file:///home/manifoldismay/Documents/Projects/LatentHackers/test.ipynb): A Jupyter notebook demonstrating how to:
  - Load the dwelling geometry dataset using `pandas` and `geopandas`.
  - Parse WKT (Well-Known Text) spatial data.
  - Buffer room geometries to generate solid apartment outlines.
  - Visualize and compare input outlines versus target room partitions using `matplotlib`.
- `data/modified-swiss-dwellings/mds_V2_5.372k.csv`: Local copy of the Modified Swiss Dwellings dataset.
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

## Baseline Data Pipeline

Install the baseline dependencies in one environment:

```bash
pip install -r requirements.txt
```

Compute dataset statistics:

```bash
python scripts/dataset_stats.py
```

Write the dataset-provided train/test split, mapped from `floor_id` filenames to `plan_id`:

```bash
python scripts/write_splits.py --output outputs/splits.json
```

The original dataset does not provide validation IDs. To carve a deterministic validation subset from the original training set, pass `--val-fraction`, for example `--val-fraction 0.1`.

Materialize normalized arrays, fixed-length room polygon vertices, and inverse transforms. The default `max_rooms` is 160, which is just above the 99th percentile room count; the script reports how many plans are truncated.

```bash
python scripts/prepare_dataset.py \
  --output-dir outputs/prepared \
  --val-fraction 0.1 \
  --representation partition \
  --vertex-count 16 \
  --seed 42
```

Train the small conditional flow-matching Transformer:

```bash
python scripts/train_flow.py \
  --prepared-dir outputs/prepared \
  --output-dir outputs/flow_baseline \
  --epochs 100 \
  --batch-size 8 \
  --lr 0.0001 \
  --d-model 256 \
  --nhead 8 \
  --encoder-layers 4 \
  --decoder-layers 4 \
  --ffn-dim 1024 \
  --cond-dim 256 \
  --seed 42
```

Resume training:

```bash
python scripts/train_flow.py \
  --prepared-dir outputs/prepared \
  --output-dir outputs/flow_baseline \
  --resume outputs/flow_baseline/last.pt \
  --epochs 100 \
  --seed 42
```

Sample generated layouts from a checkpoint:

```bash
python scripts/sample_flow.py \
  --checkpoint outputs/flow_baseline/best.pt \
  --plan-id 1054 \
  --compare-plan-id 5945 \
  --output-dir outputs/samples/1054 \
  --num-samples 4 \
  --steps 32 \
  --seed 42
```

Sampling writes per-sample metrics before and after repair, plus diversity across samples and an optional outline-response check.

Run local raster metrics, including FID plus PRDC-style density and coverage:

```bash
python scripts/evaluate_flow.py \
  --checkpoint outputs/flow_baseline/best.pt \
  --prepared-dir outputs/prepared \
  --split val \
  --output-dir outputs/eval/val \
  --num-plans 128 \
  --samples-per-plan 4 \
  --resolution 64 \
  --feature-dim 128 \
  --knn 5 \
  --seed 42
```

This writes `metrics_summary.json`, `per_sample_metrics.csv`, `metric_distributions.png`, and best/medium/worst real-vs-generated comparison images. The organiser score is still the source of truth because their held-out reference set and rasterisation protocol are fixed externally.

Visualize one plan:

```bash
python scripts/visualize_plan.py --plan-id 1054 --output outputs/plan_1054.png
```
