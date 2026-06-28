# LatentHackers

Floor-plan generation experiments for the Modified Swiss Dwellings dataset.

The current notebooks build a simple outline-conditioned baseline, store the trained model, and evaluate generated layouts on the official held-out test split.

## Environment

Use the `ehl` conda environment.

Freeze the current environment:

```bash
conda env export -n ehl --from-history > environment.yml
conda env export -n ehl > environment.full.yml
```

Recreate it:

```bash
conda env create -n ehl_copy -f environment.yml
```

If dependencies are missing, install them inside `ehl`:

```bash
conda activate ehl
pip install -r requirements.txt
```

## Data

The notebooks expect the MSD CSV at the repository root:

```text
mds_V2_5.372k.csv
```

The official split is read from:

```text
train_indices.txt
test_indices.txt
```

The evaluation notebook uses the test split by default.

## Main Notebook

Run:

```text
main.ipynb
```

It:

- loads and explores the MSD room geometries;
- constructs apartment outlines using the `0.3 m` buffer-union-buffer method;
- builds padded vector room representations;
- defines the outline-conditioned denoiser baseline;
- trains or loads the model;
- stores the model checkpoint.

Stored model:

```text
artifacts/outline_conditioned_denoiser.pt
```

The checkpoint includes model weights, config, room-type mappings, split IDs, and training history.

## Dataset Evaluation Notebook

Run:

```text
dataExploration/data_exploration.ipynb
```

The model-loading cell imports definitions from `main.ipynb` without retraining or running demo plots.

Default evaluation settings:

```python
EVALUATION_SPLIT = "test"
EVALUATION_LIMIT = None
```

Supported diagnostic splits:

```python
EVALUATION_SPLIT = "train"
EVALUATION_SPLIT = "val"
EVALUATION_SPLIT = "all"
```

Use a small smoke run while iterating:

```python
EVALUATION_LIMIT = 128
```

The evaluation cell writes test-specific outputs:

```text
artifacts/dataset_metrics/test_per_plan_model_metrics.csv
artifacts/dataset_metrics/test_generation_failures.csv
artifacts/dataset_metrics/test_summary_model_metrics.json
artifacts/dataset_metrics/test/real/
artifacts/dataset_metrics/test/fake/
```

The image-metric cell computes:

- FID;
- density;
- coverage.

It writes:

```text
artifacts/dataset_metrics/test_image_metric_summary.json
artifacts/dataset_metrics/test_real_inception_features.npy
artifacts/dataset_metrics/test_fake_inception_features.npy
```

## Speed Controls

`data_exploration.ipynb` caches intermediate outputs.

Regenerate layouts:

```python
FORCE_REGENERATE_LAYOUTS = True
```

Rerender images:

```python
FORCE_RERENDER_IMAGES = True
```

Recompute cached Inception features:

```python
FORCE_RECOMPUTE_IMAGE_FEATURES = True
```

Skip rendering when only geometric metrics are needed:

```python
RENDER_IMAGES = False
```

Limit image metrics:

```python
IMAGE_METRIC_LIMIT = 512
```

## Important Notes

- Do not commit `.codex/`, local caches, generated images, or large model artifacts unless explicitly needed.
- The stored notebook checkpoint is a lightweight baseline artifact, not a final trained competition model.
- `data_exploration.ipynb` evaluates the test split by default; use `all` only for diagnostics.

## FlowMatching2

`FlowMatching2/` contains the more modular flow-matching and wall-graph experiments. Its wall-graph v1 path supports:

- full wall-graph preprocessing;
- junction-coordinate flow matching;
- junction presence prediction;
- dense edge prediction;
- deterministic validation;
- threshold-sweep evaluation.

Prepared full wall-graph data is expected at:

```text
FlowMatching2/outputs/prepared_wall_graph
```

Recommended wall-graph training command from inside `FlowMatching2/`:

```bash
python train.py \
  --prepared-dir outputs/prepared_wall_graph \
  --output-dir outputs/wall_graph_v1 \
  --geometry-representation wall_graph \
  --epochs 150 \
  --batch-size 8 \
  --lr 0.0001 \
  --d-model 256 \
  --nhead 8 \
  --encoder-layers 4 \
  --decoder-layers 4 \
  --ffn-dim 1024 \
  --lambda-junction-flow 1.0 \
  --lambda-junction-endpoint 1.0 \
  --lambda-junction-presence 0.2 \
  --lambda-edge 1.0 \
  --edge-loss bce \
  --edge-pos-weight-max 50 \
  --wall-graph-mask-warmup-epochs 20 \
  --wall-graph-mask-transition-epochs 20 \
  --device cuda \
  --seed 42 \
  --progress-every 25
```

Evaluate:

```bash
python scripts/evaluate_wall_graph.py \
  --checkpoint outputs/wall_graph_v1/best.pt \
  --prepared-dir outputs/prepared_wall_graph \
  --split val \
  --output-dir outputs/wall_graph_v1_eval \
  --num-plans 128 \
  --steps 100 \
  --seed 42 \
  --device cuda \
  --edge-thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7
```
