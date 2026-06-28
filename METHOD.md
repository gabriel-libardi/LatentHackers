# Methodology and Iteration Notes

## Challenge Framing

The task is conditional vector floor-plan generation. Given only the outline of an apartment, the model must generate a complete set of typed room polygons. The output must remain vector geometry, not a pixel grid.

The useful data source is `mds_V2_5.372k.csv`. Room polygons are selected with:

```python
entity_type == "area"
```

For each `plan_id`, the condition outline is built from the room polygons by:

1. buffering every room outward by `0.3 m`,
2. unioning the buffered rooms,
3. buffering inward by `0.3 m`.

This follows the challenge brief and gives a single exterior shell as the model condition.

## Initial Issues Found

The first notebook version produced results that looked almost identical to ground truth. That was misleading for several reasons.

The original split was a random split over plans. Many validation plans had near-duplicate train plans from the same site or building, so nearest-neighbour retrieval could copy a sibling layout.

The first prediction metric used only the union footprint IoU. This ignores internal room partitioning and labels, so a prediction could cover the same exterior area while having the wrong room count or wrong room types.

The first neural model stub denoised coordinates but did not meaningfully condition on the apartment outline. It also assumed target room types and masks were already available, which is not the real challenge setting.

## Split Improvement

The notebook now uses a deterministic site-level holdout split as a safer validation proxy. All plans from one `site_id` stay in the same split.

This reduces leakage from near-identical plans in the same building cluster and makes validation samples less artificially easy.

The official challenge split still exists in `modified-swiss-dwellings-v2/{train,test}`, but those example IDs are not directly mapped back to the CSV geometry workflow in the notebook. The site-level split is therefore used as a practical leakage-resistant notebook split.

## Vector Representation

The representation follows the useful parts of HouseDiffusion:

- each room is a polygon loop,
- polygon exterior coordinates are quantized to an 8-bit grid `[0, 255]`,
- quantized coordinates are mapped to continuous `[-1, 1]` values for diffusion,
- integer coordinate bits are also stored for a later discrete denoising head.

The training tensors are padded:

```text
coords[room, corner, xy]
grid[room, corner, xy]
bits[room, corner, xy, bit]
corner_mask[room, corner]
room_mask[room]
roomtype_ids[room]
outline_cont[outline_corner, xy]
outline_mask[outline_corner]
```

This keeps the output vector-native while still fitting fixed-shape neural models.

## Baseline Improvements

### 1. Leakage-Aware Retrieval

The nearest-neighbour baseline was changed so candidates from the same site or building can be excluded. This prevents the baseline from simply retrieving a near-duplicate sibling plan.

### 2. Diversity-Aware Sampling

Instead of always selecting the single closest outline, the notebook samples multiple distinct candidates from a ranked candidate set. This better reflects the problem: one apartment outline can admit several plausible room configurations.

### 3. Program-Aware Scoring

The retrieval baseline was further improved with a conditional room-program prior. For a query outline, the notebook estimates likely:

- room count,
- room-type counts,
- target coverage ratio.

It then scores candidate layouts using:

- outline similarity,
- room-count error against the prior,
- room-type count error against the prior,
- outline coverage error against the prior,
- room naturalness penalty based on training-set room area and compactness statistics.

This makes the baseline much more stable. For example, a validation outline with an estimated program of about `53` rooms now samples candidates near that range instead of wildly varying between very small and very large layouts.

### 4. Better Diagnostics

The notebook now reports partition-aware metrics:

- `room_count_error`,
- `roomtype_count_l1`,
- `mean_best_same_type_iou`,
- `prediction_outline_coverage`,
- `union_iou`.

`union_iou` is still useful, but it is no longer treated as sufficient on its own.

## Neural Model Improvements

### 1. Outline-Conditioned Denoiser

The original `TinyVectorDenoiser` was replaced by `OutlineConditionedVectorDenoiser`.

The new model:

- encodes padded outline tokens,
- pools the outline into a condition vector,
- injects the outline condition into every room-corner token,
- uses room and corner positional embeddings,
- predicts coordinate noise,
- predicts coordinate bits,
- predicts room types.

This aligns the architecture with the actual challenge input: the model receives an outline and must generate typed room polygons.

### 2. Real Diffusion Training Step

The notebook now includes a real training-step contract:

```text
loss = noise_loss + bit_loss_weight * bit_loss + roomtype_loss_weight * roomtype_loss
```

The losses are:

- `noise_loss`: MSE between predicted and true diffusion noise,
- `bit_loss`: binary cross-entropy on quantized coordinate bits,
- `roomtype_loss`: cross-entropy on room-type labels.

The model cell was validated end-to-end and produced a real loss dictionary.

### 3. Reduced Room-Type Teacher Forcing

An issue was found in the first denoiser version: during training the model received the true `roomtype_ids` as input and was also asked to predict room types. That makes the room-type head partly teacher-forced, so it can look valid in a smoke test while not learning a robust outline-conditioned room-type prior.

The training step now applies room-type condition dropout. For a random subset of valid rooms, the input room type is replaced with the padding/unknown id while the loss is still computed against the true room type. This forces the model to use outline geometry, coordinate context, and room positions instead of only copying the provided room-type embedding.

### 4. Training Loop and Checkpoint

A new training section was added:

```python
TRAINING_MAX_STEPS = 8
TRAINING_BATCH_SIZE = 2
TRAINING_SUBSET_SIZE = 192
CHECKPOINT_PATH = "artifacts/outline_conditioned_denoiser.pt"
```

The loop trains the outline-conditioned denoiser for a small configurable number of steps, clips gradients, and saves a checkpoint.

The current checkpoint path is:

```text
artifacts/outline_conditioned_denoiser.pt
```

The default step count is intentionally low so the notebook remains fast to execute. For better quality, increase `TRAINING_MAX_STEPS` to `100`, then `500+` if runtime allows.

## Scaffold-Then-Refine Sampling

The current generation path combines the strongest baseline with conservative geometry repair. The diffusion model is kept as a diagnostic refiner for now, because the current short checkpoint is not trained enough to produce natural-looking final polygons.

1. The program-aware vector baseline proposes a plausible scaffold layout.
2. The scaffold is converted to padded coordinate tensors.
3. The scaffold is clipped to the target outline.
4. Missing regions inside the outline are assigned to adjacent existing rooms.
5. The repaired polygons are plotted as the final candidate.
6. The diffusion denoiser can optionally be enabled as a diagnostic residual refiner, but it is not used by default for final display.
7. Scaffold and final outputs are compared with the same metrics.

This is not yet a full unconditional reverse diffusion sampler. It is a practical intermediate step: use a strong vector scaffold, repair the most obvious geometric gaps, and keep the learned denoiser separate until it has enough training to improve rather than damage the layout.

### Refinement Issue and Fix

The first refinement pass was too aggressive. It added noise at a relatively large timestep, trusted a very lightly trained denoiser to reconstruct clean coordinates, and replaced scaffold room labels with model-predicted labels. In practice, this can damage a good scaffold, especially after only a few training steps.

The refinement path is now conservative:

- default denoising timestep was reduced,
- predicted clean coordinates are used as a residual update instead of a full replacement,
- scaffold room types are preserved by default,
- refinement is rejected if coverage drops meaningfully or if too many rooms disappear.

This makes the learned model act as a geometric refiner while the stronger program-aware baseline remains responsible for room count and room-type scaffolding until the model has been trained long enough to take over more of the generation process.

### Why Refined Looked Identical

After adding the guard, the denoiser-refined panel could look identical to the program scaffold. That was not a plotting bug. With a very small checkpoint, the denoiser proposal was often rejected by the guard, so the function returned the original scaffold unchanged.

The first fallback added standalone corridor-like filler polygons for missing regions. That made the panel visibly different, but the result could look unnatural: large artificial corridor blobs are not a good architectural prior.

The generation path now uses a more conservative fallback:

- uncovered gaps are assigned to adjacent existing rooms whenever possible,
- standalone filler rooms are disabled by default,
- the repaired geometry is clipped to the outline,
- room overlaps are removed before gap assignment,
- gap assignment is capped so rooms cannot grow unrealistically just to fill space,
- coverage and cleanup diagnostics are reported,
- the diffusion proposal is optional and gated behind `neural_refinement_enabled=True`.

This means the final panel should visibly differ from the scaffold when there are repairable gaps, should usually improve outline coverage, and should avoid both destructive neural edits and artificial filler blobs from an undertrained checkpoint.

### Geometry Cleanup Pass

The coverage repair was improved again because assigning gaps alone can create rough geometry or preserve overlaps from the scaffold. The final post-processing now does three steps:

1. Clip all rooms to the target outline.
2. Resolve overlaps by assigning already-claimed area to larger rooms first.
3. Iteratively assign remaining outline gaps to adjacent rooms.

The comparison table now includes cleanup diagnostics:

- `coverage_added_area_m2`,
- `remaining_gap_area_m2`,
- `overlap_removed_area_m2`.
- `scaffold_naturalness_penalty`,
- `refined_naturalness_penalty`.

These are no-ground-truth metrics, so they can be used on hidden/test outlines too. They make it easier to see whether the final layout actually fills the outline and whether cleanup had to remove a lot of overlapping geometry.

### More Visual Examples

The comparison plot now shows four generated candidates by default instead of two. This makes it easier to inspect diversity and identify whether a failure is systematic or only one bad sample.

The default final call is:

```python
comparison_df = compare_scaffold_and_refined(n_samples=4, neural_refinement_enabled=ENABLE_NEURAL_REFINEMENT)
```

### Diffusion Output Was Gated Off

The diffusion output currently does not look natural enough to be used as the final sample. The main reason is not the architecture alone, but the training regime: the checkpoint is extremely small, so coordinate denoising has not learned a stable geometric prior yet.

The notebook therefore separates responsibilities:

- program-aware retrieval proposes the room program and rough layout,
- deterministic repair improves outline coverage,
- diffusion refinement remains available as an optional diagnostic path,
- final displayed samples use `neural_refinement_enabled=False` by default.

The notebook exposes this explicitly:

```python
RUN_DENOISER_TRAINING = False
ENABLE_NEURAL_REFINEMENT = False
```

Set `RUN_DENOISER_TRAINING=True` when you want to update the checkpoint. Set `ENABLE_NEURAL_REFINEMENT=True` only after a checkpoint produces natural geometry in diagnostics.

Once longer training produces stable refinements, the diagnostic denoiser can be re-enabled and compared against the repaired scaffold with the same metrics.

## Current Status

The notebook now contains:

- leakage-resistant data splitting,
- vector-native room representation,
- program-aware retrieval baseline,
- stronger evaluation diagnostics,
- outline-conditioned denoiser,
- room-type condition dropout,
- real diffusion training loss,
- checkpointing,
- conservative scaffold-refinement sampling and comparison,
- coverage repair for missing regions inside the outline,
- overlap cleanup and geometric repair diagnostics,
- naturalness-aware candidate ranking and repair limits,
- four-sample comparison plots by default,
- diffusion refinement gated off by default until trained checkpoints produce natural geometry.

The pipeline is now trainable and testable end to end. The biggest remaining quality gains should come from longer training, better generation of room masks/counts without relying on the scaffold, and a fuller reverse diffusion sampler.

## Recommended Next Improvements

1. Increase training duration and track validation losses.
2. Add a learned room-program head to predict room count and room-type counts directly from the outline.
3. Replace scaffold refinement with full reverse diffusion sampling.
4. Add geometric post-processing to enforce shared walls, remove overlaps, and fill gaps.
5. Evaluate samples using the challenge-style rasterized FID, density, and coverage pipeline once the organisers' evaluation protocol is available.
