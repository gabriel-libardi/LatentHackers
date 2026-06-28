# HouseDiffusion 2: Adapted Outline-Conditioned HouseDiffusion

This repository contains the implementation of **HouseDiffusion 2**, an adapted version of the HouseDiffusion architecture designed to generate vector room layouts conditioned exclusively on the **outer apartment outline** rather than a constraint graph (bubble diagram).

---

## 1. Core Differences: HouseDiffusion vs. HouseDiffusion 2

The table below outlines how this adapted model (**HouseDiffusion 2**) differs from the original **HouseDiffusion** (CVPR 2023) framework:

| Feature | Original HouseDiffusion | HouseDiffusion 2 (Adapted) |
| :--- | :--- | :--- |
| **Conditioning Input** | **Relational Graph / Bubble Diagram:** A graph specifying the room types and which rooms are connected to which doors. | **Bare Apartment Outline:** A single polygon representing the exterior boundary of the apartment. No relational graph or connectivity information is provided. |
| **Attention Mechanism** | **CSA, GSA, and RCA:** Uses Component-wise Self-Attention (CSA), Global Self-Attention (GSA), and Relational Cross-Attention (RCA) to enforce graph connectivity constraints. | **CSA, GSA, and OCA:** Replaces RCA with **Outline Cross-Attention (OCA)**, allowing room corners to attend to and align with the outer boundary vertices. |
| **Query Structure** | **Graph-Specified:** The number of rooms, room types, and corners per room are explicitly given by the input graph. | **Retrieved Dynamically:** Resolves the lack of input graph by matching the input outline to the training set and retrieving a matching room layout template. |
| **Coordinate Space** | **Global Dataset Scale:** Normalization is done globally or per plan without strict reference to a bounding shell. | **Outline Bounding Box Scale:** Normalizes all coordinates (rooms and outline) to $[0, 1]^2$ based on the outline's bounding box, preserving the exact aspect ratio. |

---

## 1.1 Apartment Contour Representation & Encoding

To feed the apartment outline (the conditioning input) into our model, we use the following representation and encoding pipeline:

1. **Extraction & Geometry Processing:**
   We obtain the outline polygon by taking the union of all room polygons buffered outward by 0.3 meters and then buffered back inward by 0.3 meters. This fuses individual rooms into a clean, continuous outer boundary shell.
2. **Vertex Extraction & Sequential Ordering:**
   We extract the list of 2D coordinates $(x_i, y_i)$ of the exterior boundary vertices. The points are ordered sequentially (clockwise or counterclockwise) along the perimeter of the apartment shell.
3. **Bounding-Box Normalization:**
   To align with the coordinate space of the diffusion room corners, all outline coordinates are normalized relative to the outline's bounding box:
   $$x_{\text{norm}} = \frac{x - \text{minx}}{\text{max\_size}}, \quad y_{\text{norm}} = \frac{y - \text{miny}}{\text{max\_size}}$$
   This maps the outline coordinates into the range $[0, 1]^2$ while preserving the exact aspect ratio of the apartment.
4. **Uniform Downsampling & Padding:**
   - If the outline has more than $M_{\text{max}} = 128$ vertices, we uniformly downsample the points along the perimeter to fit the model constraint.
   - If the outline has fewer than 128 vertices, we pad it to 128 with zero-vectors and use a binary `outline_mask` (of shape `(B, M)`) where valid vertices are set to `1.0` and padded vertices to `0.0`.
5. **Continuous & Positional Encoding:**
   The outline tensor (shape `(B, 128, 2)`) is fed into the `OutlineEncoder` network, which:
   - Projects the 2D coordinates to a continuous space of dimension `d_model` via a linear projection.
   - Adds 1D learnable position embeddings corresponding to each index ($0$ to $127$) to preserve the sequential connectivity order of the vertices along the perimeter.
   - Passes the sum through a SiLU-activated multi-layer MLP to obtain outline token embeddings $z^{\text{outline}} \in \mathbb{R}^{B \times 128 \times d_{\text{model}}}$.
6. **Cross-Attention Conditioning:**
   These tokens are fed as Key/Value representations to the **Outline Cross-Attention (OCA)** blocks in each layer of the Transformer, enabling query room corner tokens to attend to and align with the apartment boundary.

---

## 1.2 Corner Augmentation (AU): Wall Segment Sampling

To provide the Transformer self-attention layers with rich, explicit geometric context about the walls connecting room corners, we implement **Corner Augmentation (AU)**:
1. **Wall Vectorization:** For every corner coordinate $x_i$, we identify its successor $x_{i, \text{next}}$ in the room polygon sequence (wrapping around to the room's first corner token for the final boundary vertex).
2. **Intermediate Sampling:** We uniformly sample 8 point coordinates along the connecting wall segment:
   $$p_k = (1 - \lambda_k) x_i + \lambda_k x_{i, \text{next}}, \quad \lambda_k \in \text{linspace}(0.0, 1.0, 8)$$
3. **Concatenation:** We concatenate the 8 sampled points (16 values) directly with the original corner coordinates (2 values) to form an augmented 18-dimensional feature representation for each token before projecting it to `d_model` via a linear embedding layer.

## 1.3 Discrete Denoising Branch & Dynamic Snapping

To ensure adjacent rooms share clean boundary alignments (avoiding minor offsets or gaps), we implement a dual-branch output head:
1. **Continuous Branch:** Regresses the continuous noise vector $\epsilon_\theta \in \mathbb{R}^2$ at all timesteps to steer the diffusion reverse process.
2. **Discrete Branch:** Predicts an 8-bit binary representation (16-dimensional logit vector representing integers $[0, 255]$ for $x$ and $y$ coordinates) of the *final clean room corners* $x_0$.
3. **Combined Training Loss:** The model optimizes the sum of L2 losses on both branches:
   $$\mathcal{L} = \mathcal{L}_{\text{continuous}} + \mathcal{L}_{\text{discrete}}$$
4. **Dynamic Coordinate Snapping:** During sampling inference, when the timestep $t$ falls below a threshold ($T_{\text{threshold}} = 32$), the model reconstructs the $x_0$ estimate directly from the discrete branch predictions to dynamically snap coordinates into clean alignment.

## 1.4 Probabilistic Prior Template Matching

At inference time, the model is only provided with a bare outline. To resolve room configurations (number of rooms, category ids, and corners per room) without an input constraint graph, we implement a **probabilistic prior matching** mechanism:
1. **Centroid-Aligned IoU Database:** We search the training outlines database and score candidates using centroid-aligned Intersection over Union (IoU).
2. **Probabilistic Selection:** Instead of selecting the single closest match deterministically, the generator identifies the **top-10** best-fitting templates and randomly samples one. This probabilistically samples the room configurations from the empirical training distribution of similarly shaped apartments.

---

## 2. Step-by-Step Methodology

### Step 1: Split Index Extraction (`extract_indices.py`)
Because the dataset is provided as a single large CSV (`mds_V2_5.372k.csv`), we extract the training and testing partition splits by parsing the filenames inside the `modified-swiss-dwellings-v2` splits directory:
- We match the filenames (which are `floor_id` values) to separate the dataset into 4,572 training plans and 800 testing plans.
- Output text files ([train_indices.txt](train_indices.txt) and [test_indices.txt](test_indices.txt)) contain the clean, sorted IDs.

### Step 2: Data Preprocessing & Normalization (`dataset.py`)
For a given floor plan:
1. **Outline Fusion:** We load room polygons where `entity_type == area`. We buffer them outward by 0.3m, unify them into a single shape, and buffer back inward by -0.3m to fuse them into a single exterior shell.
2. **Aspect-Ratio Normalization:** We compute the bounding box of the outline and normalize both the outline vertices and the room corner coordinates:
   $$x_{\text{norm}} = \frac{x - \text{minx}}{\text{max\_size}}, \quad y_{\text{norm}} = \frac{y - \text{miny}}{\text{max\_size}}$$
3. **Sequence Collating:** Different plans have varying room and outline lengths. We collate them into batch tensors padded with zero-tokens, outputting sequence masks (`corner_mask`, `outline_mask`) to let the attention layers ignore padding.

### Step 3: Diffusion Training (`train.py`)
We train a Transformer-based diffusion model:
1. **Forward Process:** At each training step, we sample random timesteps $t \in [0, 999]$ and add Gaussian noise to the normalized room coordinates using the cosine schedule.
2. **Model Prediction:** The model takes noisy coordinates, entity types, entity IDs, corner indices, outline coordinates, and timesteps. It outputs predicted noise $\epsilon_\theta$.
3. **Masked Loss:** We compute Mean Squared Error (MSE) loss between the predicted noise and the actual noise, masking out loss on the padded corner tokens. We optimize using AdamW with gradient clipping.

### Step 4: Shape-Based Query Retrieval (`val.py`)
At inference, we are given only the bare outline polygon. Since we do not have room counts or categories:
1. We compute a **centroid-aligned Intersection over Union (IoU)** between the query outline and the training outlines database.
2. We select the training outline with the highest shape similarity and retrieve its room configuration (room categories, instance IDs, and corner counts per room) to initialize the query template for the diffusion model.
3. This guarantees we generate a highly realistic set of rooms matching the size and shape of the apartment.

### Step 5: Reverse Denoising & Reconstruction (`val.py`)
1. **Denoising Loop:** Starting from random Gaussian noise, the model iteratively predicts noise and updates the coordinates for 1,000 steps, guided by Outline Cross-Attention.
2. **De-normalization:** We map the final coordinates back to the original coordinate space using the query outline's bounding box parameters.
3. **Polygon Fitting:** We group the coordinates by room ID, sort them by corner index, and construct the final Shapely Polygons.
4. **Rendering:** Results are plotted side-by-side (Input Outline, Predicted Layout, Ground Truth Layout) and saved as images in `val_results/`.

---

## 3. Usage Instructions

### Dependencies
Ensure the packages in [requirements.txt](requirements.txt) are installed:
```bash
pip install -r requirements.txt
```

### 1. Training the Model
To start training on GPUs:
```bash
python3 train.py --epochs 50 --batch_size 16 --lr 2e-4
```

### 2. Validation & Layout Generation
To run validation and output comparative visualization plots:
```bash
python3 val.py --model_path model_final.pth
```
Comparative plots will be saved in `val_results/`.
