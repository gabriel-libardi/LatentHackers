# HouseFlowMatching

Outline-conditioned **flow matching** model for vector floorplan generation on the Modified Swiss Dwellings (MSD) dataset.

This is the flow-matching counterpart of our diffusion model (`HouseDiffusion2`). It shares the **exact same Transformer backbone** (component-wise self-attention, global self-attention, outline cross-attention) and the **same data pipeline**. Only the generative process differs:

- **Diffusion (DDPM):** the network predicts the noise added to room-corner coordinates and denoises over 1000 discrete steps.
- **Flow matching (this folder):** the network predicts a **velocity field** along a straight (optimal-transport) path between noise and data, and generation integrates an ODE in ~100 Euler steps — smoother trajectories and much faster sampling.

Given only an apartment outline, the model fills in the interior rooms as typed vector polygons.

## Files

- `dataset.py` — MSD loader: builds the outline (buffer +0.3 / union / buffer −0.3), normalizes coordinates, extracts room corners with room-type / room-instance / corner indices.
- `house_flow.py` — Transformer backbone (`HouseFlowModel`) and the `FlowMatching` helper (path, velocity target, ODE sampler).
- `train.py` — training loop (rectified-flow velocity objective, seed 42).
- `val.py` — generation + qualitative comparison plots.
- `requirements.txt`

## Method in two lines

With `data` = clean corners and `noise ~ N(0, I)`:

```
x_t = (1 - t) * noise + t * data          # straight path,  t in [0, 1]
target_velocity = data - noise            # constant velocity
loss = MSE( model(x_t, t, outline) , target_velocity )
```

Sampling starts from `noise` and integrates `dx/dt = model(x, t, outline)` from `t = 0` to `t = 1`.

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py --csv_path mds_V2_5.372k.csv --train_index train_indices.txt --epochs 50 --batch_size 16
```

Checkpoints are saved as `flow_epoch_*.pth` and `flow_final.pth`.

## Generate / validate

```bash
python val.py --csv_path mds_V2_5.372k.csv --train_index train_indices.txt --test_index test_indices.txt --model_path flow_final.pth
```

The entry point `OutlineConditionedGenerator(...).generate(outline_polygon)` returns a list of `(shapely.Polygon, room_type)` tuples.

## Notes

- A fixed random seed of **42** is used across data, training and sampling, as required by the challenge.
- Flow matching reaches quality comparable to the diffusion model while sampling in ~100 ODE steps instead of 1000, which is convenient for generating many samples (useful for the diversity / coverage metric).
- Backbone, hyper-parameters (`d_model=256`, `num_heads=8`, `d_ff=1024`, `num_layers=6`, `max_corners_per_room=512`, `max_rooms=512`, `max_outline_len=128`) and the retrieval-based room structure are identical to the diffusion model, so the two are directly comparable in an ablation.
