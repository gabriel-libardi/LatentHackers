from pathlib import Path
import os

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from IPython.display import display
except ImportError:
    def display(*args, **kwargs):
        for arg in args:
            print(arg)


pd.set_option("display.max_columns", 80)
pd.set_option("display.max_colwidth", 120)

SEED = 42
CSV_PATH = Path("mds_V2_5.372k.csv")
WALL_BRIDGE_DISTANCE_M = 0.30
rng = np.random.default_rng(SEED)

raw_df = pd.read_csv(CSV_PATH)
print(f"rows: {len(raw_df):,}")
print(f"columns: {len(raw_df.columns)}")
display(raw_df.head())
display(raw_df.dtypes.rename("dtype").to_frame())

summary = pd.DataFrame({
    "missing": raw_df.isna().sum(),
    "missing_pct": raw_df.isna().mean().mul(100).round(2),
    "n_unique": raw_df.nunique(dropna=True),
    "dtype": raw_df.dtypes.astype(str),
})
display(summary)

id_cols = ["site_id", "building_id", "floor_id", "plan_id", "apartment_id", "unit_id", "area_id"]
entity_counts = raw_df["entity_type"].value_counts(dropna=False).rename_axis("entity_type").to_frame("rows")
subtype_counts = raw_df["entity_subtype"].value_counts(dropna=False).rename_axis("entity_subtype").to_frame("rows")
roomtype_counts = raw_df["roomtype"].value_counts(dropna=False).rename_axis("roomtype").to_frame("rows")
unique_ids = raw_df[id_cols].nunique(dropna=True).rename("n_unique").to_frame()

display(entity_counts)
display(subtype_counts.head(30))
display(roomtype_counts)
display(unique_ids)

area_df = raw_df.loc[raw_df["entity_type"].eq("area")].copy()
area_df["geometry"] = area_df["geom"].map(wkt.loads)
rooms_gdf = gpd.GeoDataFrame(area_df, geometry="geometry")

rooms_gdf["is_valid"] = rooms_gdf.geometry.is_valid
rooms_gdf["is_empty"] = rooms_gdf.geometry.is_empty
rooms_gdf["area_m2"] = rooms_gdf.geometry.area
rooms_gdf["perimeter_m"] = rooms_gdf.geometry.length
rooms_gdf["bounds_tuple"] = rooms_gdf.geometry.bounds.apply(tuple, axis=1)
rooms_gdf[["minx", "miny", "maxx", "maxy"]] = rooms_gdf.geometry.bounds
rooms_gdf["width_m"] = rooms_gdf["maxx"] - rooms_gdf["minx"]
rooms_gdf["height_m"] = rooms_gdf["maxy"] - rooms_gdf["miny"]
rooms_gdf["n_exterior_vertices"] = rooms_gdf.geometry.map(
    lambda geom: len(geom.exterior.coords) if geom.geom_type == "Polygon" else np.nan
)

quality = pd.Series({
    "area_rows": len(rooms_gdf),
    "plans_with_rooms": rooms_gdf["plan_id"].nunique(),
    "invalid_geometries": int((~rooms_gdf["is_valid"]).sum()),
    "empty_geometries": int(rooms_gdf["is_empty"].sum()),
    "zero_or_negative_area": int((rooms_gdf["area_m2"] <= 0).sum()),
})
display(quality.to_frame("value"))
display(rooms_gdf[["plan_id", "entity_subtype", "roomtype", "area_m2", "perimeter_m", "n_exterior_vertices", "geometry"]].head())

numeric_features = ["area_m2", "perimeter_m", "width_m", "height_m", "n_exterior_vertices"]
display(rooms_gdf[numeric_features].describe(percentiles=[.01, .05, .25, .50, .75, .95, .99]).T)

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
rooms_gdf["roomtype"].value_counts().plot.bar(ax=axes[0, 0], color="#5b8db8")
axes[0, 0].set_title("Room/entity labels")
axes[0, 0].set_ylabel("rows")

rooms_gdf["area_m2"].clip(upper=80).hist(ax=axes[0, 1], bins=60, color="#6aa56a")
axes[0, 1].set_title("Room area distribution clipped at 80 m2")
axes[0, 1].set_xlabel("m2")

rooms_gdf["n_exterior_vertices"].clip(upper=40).hist(ax=axes[1, 0], bins=40, color="#b58b4a")
axes[1, 0].set_title("Exterior vertex count clipped at 40")
axes[1, 0].set_xlabel("vertices")

rooms_gdf.plot.scatter("width_m", "height_m", s=2, alpha=0.08, ax=axes[1, 1], color="#7c6aa8")
axes[1, 1].set_title("Room bounding-box dimensions")
axes[1, 1].set_xlabel("width m")
axes[1, 1].set_ylabel("height m")
axes[1, 1].set_xlim(0, rooms_gdf["width_m"].quantile(.995))
axes[1, 1].set_ylim(0, rooms_gdf["height_m"].quantile(.995))

plt.tight_layout()
plt.savefig("room_distributions.png")
plt.close(fig)

plan_bounds = rooms_gdf.groupby("plan_id")[["minx", "miny", "maxx", "maxy"]].agg({
    "minx": "min", "miny": "min", "maxx": "max", "maxy": "max"
})
plan_stats = rooms_gdf.groupby("plan_id").agg(
    room_count=("geometry", "size"),
    total_area_m2=("area_m2", "sum"),
    mean_room_area_m2=("area_m2", "mean"),
    max_room_area_m2=("area_m2", "max"),
    unique_roomtypes=("roomtype", "nunique"),
).join(plan_bounds)
plan_stats["extent_width_m"] = plan_stats["maxx"] - plan_stats["minx"]
plan_stats["extent_height_m"] = plan_stats["maxy"] - plan_stats["miny"]
plan_stats["extent_area_m2"] = plan_stats["extent_width_m"] * plan_stats["extent_height_m"]
roomtype_counts_by_plan = {
    plan_id: counts.to_dict()
    for plan_id, counts in rooms_gdf.groupby("plan_id")["roomtype"].value_counts().groupby(level=0)
}
plan_stats["roomtype_counts"] = plan_stats.index.map(roomtype_counts_by_plan)

display(plan_stats.describe(percentiles=[.01, .05, .25, .50, .75, .95, .99]).T)
display(plan_stats.sort_values("room_count").head())
display(plan_stats.sort_values("room_count", ascending=False).head())

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
plan_stats["room_count"].hist(ax=axes[0], bins=60, color="#5b8db8")
axes[0].set_title("Rooms per plan")
axes[0].set_xlabel("room count")

plan_stats["total_area_m2"].clip(upper=1500).hist(ax=axes[1], bins=60, color="#6aa56a")
axes[1].set_title("Total room area per plan, clipped")
axes[1].set_xlabel("m2")

plan_stats.plot.scatter("extent_width_m", "extent_height_m", s=8, alpha=.25, ax=axes[2], color="#7c6aa8")
axes[2].set_title("Plan extents")
axes[2].set_xlabel("width m")
axes[2].set_ylabel("height m")
axes[2].set_xlim(0, plan_stats["extent_width_m"].quantile(.995))
axes[2].set_ylim(0, plan_stats["extent_height_m"].quantile(.995))

plt.tight_layout()
plt.savefig("plan_stats.png")
plt.close(fig)

room_presence = (
    rooms_gdf.assign(present=1)
    .pivot_table(index="plan_id", columns="roomtype", values="present", aggfunc="max", fill_value=0)
)
co_occurrence = room_presence.T.dot(room_presence)
room_presence_rate = room_presence.mean().sort_values(ascending=False).rename("plan_presence_rate")

display(room_presence_rate.mul(100).round(1).astype(str).add("%").to_frame())
display(co_occurrence)

def union_geometries(geometries):
    if hasattr(geometries, "union_all"):
        return geometries.union_all()
    return unary_union(list(geometries))


def build_outline(room_geometries, bridge_distance=WALL_BRIDGE_DISTANCE_M):
    buffered = room_geometries.buffer(bridge_distance)
    return union_geometries(buffered).buffer(-bridge_distance)


def get_plan_rooms(plan_id):
    return rooms_gdf.loc[rooms_gdf["plan_id"].eq(plan_id)].copy()


def get_plan_outline(plan_id, bridge_distance=WALL_BRIDGE_DISTANCE_M):
    plan_rooms = get_plan_rooms(plan_id)
    return build_outline(plan_rooms.geometry, bridge_distance=bridge_distance)

sample_plan_id = 7988 if 7988 in set(rooms_gdf["plan_id"]) else int(plan_stats.sample(1, random_state=SEED).index[0])
sample_rooms = get_plan_rooms(sample_plan_id)
sample_outline = get_plan_outline(sample_plan_id)
sample_outline_gdf = gpd.GeoDataFrame({"plan_id": [sample_plan_id]}, geometry=[sample_outline])

print(f"sample_plan_id: {sample_plan_id}")
print(f"rooms: {len(sample_rooms)}")
print(f"outline geom type: {sample_outline.geom_type}")
print(f"outline area m2: {sample_outline.area:.2f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

sample_outline_gdf.plot(ax=ax1, facecolor="#f4f4f4", edgecolor="black", linewidth=3)
ax1.set_title("Condition: clean apartment outline")
ax1.axis("equal")
ax1.axis("off")

sample_rooms.plot(ax=ax2, column="roomtype", categorical=True, legend=True, cmap="tab20", edgecolor="white", linewidth=1.2)
sample_outline_gdf.plot(ax=ax2, facecolor="none", edgecolor="black", linewidth=2, alpha=.45)
ax2.set_title("Target: typed room polygons")
ax2.axis("equal")
ax2.axis("off")

plt.suptitle(f"Training pair for plan_id={sample_plan_id}")
plt.tight_layout()
plt.savefig(f"sample_plan_pair_{sample_plan_id}.png")
plt.close(fig)

def plot_plan_pair(plan_id, ax_outline=None, ax_rooms=None):
    plan_rooms = get_plan_rooms(plan_id)
    outline = get_plan_outline(plan_id)
    outline_gdf = gpd.GeoDataFrame({"plan_id": [plan_id]}, geometry=[outline])

    if ax_outline is None or ax_rooms is None:
        _, (ax_outline, ax_rooms) = plt.subplots(1, 2, figsize=(10, 4))
    outline_gdf.plot(ax=ax_outline, facecolor="#f4f4f4", edgecolor="black", linewidth=2)
    ax_outline.set_title(f"outline {plan_id}")
    ax_outline.axis("equal")
    ax_outline.axis("off")

    plan_rooms.plot(ax=ax_rooms, column="roomtype", categorical=True, cmap="tab20", edgecolor="white", linewidth=.8)
    outline_gdf.plot(ax=ax_rooms, facecolor="none", edgecolor="black", linewidth=1.5, alpha=.45)
    ax_rooms.set_title(f"rooms {plan_id} | n={len(plan_rooms)}")
    ax_rooms.axis("equal")
    ax_rooms.axis("off")
    return ax_outline, ax_rooms

example_plan_ids = plan_stats.sample(4, random_state=SEED).index.to_list()
fig, axes = plt.subplots(len(example_plan_ids), 2, figsize=(12, 4 * len(example_plan_ids)))
for row, plan_id in enumerate(example_plan_ids):
    plot_plan_pair(plan_id, axes[row, 0], axes[row, 1])
plt.tight_layout()
plt.savefig("example_plan_pairs.png")
plt.close(fig)

from matplotlib.path import Path as MplPath

ROOMTYPE_TO_ID = {label: i + 1 for i, label in enumerate(sorted(rooms_gdf["roomtype"].unique()))}
ID_TO_ROOMTYPE = {v: k for k, v in ROOMTYPE_TO_ID.items()}


def polygon_parts(geom):
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    return []


def polygon_to_mask(geom, bounds, image_size=128):
    minx, miny, maxx, maxy = bounds
    xs = np.linspace(minx, maxx, image_size)
    ys = np.linspace(miny, maxy, image_size)
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    mask = np.zeros(points.shape[0], dtype=bool)

    for poly in polygon_parts(geom):
        exterior_path = MplPath(np.asarray(poly.exterior.coords))
        part_mask = exterior_path.contains_points(points)
        for interior in poly.interiors:
            hole_path = MplPath(np.asarray(interior.coords))
            part_mask &= ~hole_path.contains_points(points)
        mask |= part_mask
    return mask.reshape(image_size, image_size)


def plan_to_masks(plan_id, image_size=128, margin_m=1.0):
    plan_rooms = get_plan_rooms(plan_id)
    outline = get_plan_outline(plan_id)
    minx, miny, maxx, maxy = outline.bounds
    bounds = (minx - margin_m, miny - margin_m, maxx + margin_m, maxy + margin_m)

    outline_mask = polygon_to_mask(outline, bounds, image_size=image_size).astype(np.uint8)
    label_mask = np.zeros((image_size, image_size), dtype=np.uint8)
    for _, row in plan_rooms.sort_values("area_m2", ascending=False).iterrows():
        room_mask = polygon_to_mask(row.geometry, bounds, image_size=image_size)
        label_mask[room_mask] = ROOMTYPE_TO_ID[row.roomtype]
    return outline_mask, label_mask, bounds

outline_mask, label_mask, raster_bounds = plan_to_masks(sample_plan_id, image_size=128)
print("label ids:", ROOMTYPE_TO_ID)
print("outline occupancy:", outline_mask.mean().round(3))
print("target occupancy:", (label_mask > 0).mean().round(3))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(outline_mask, cmap="gray", origin="lower")
axes[0].set_title("Outline mask condition")
axes[0].axis("off")

im = axes[1].imshow(label_mask, cmap="tab20", origin="lower", vmin=0, vmax=max(ROOMTYPE_TO_ID.values()))
axes[1].set_title("Typed room mask diagnostic")
axes[1].axis("off")
plt.colorbar(im, ax=axes[1], fraction=.046, pad=.04)
plt.tight_layout()
plt.savefig(f"sample_plan_masks_{sample_plan_id}.png")
plt.close(fig)

decision_table = pd.DataFrame([
    {
        "representation": "Typed polygon set",
        "pros": "Matches required output directly; preserves geometry exactly",
        "risks": "Variable number of rooms and vertices; needs set/sequence model and validity constraints",
        "use_when": "Final architecture or post-processed output format",
    },
    {
        "representation": "Corner/vertex sequences",
        "pros": "Vector-native and compact; can condition on outline tokens",
        "risks": "Requires ordering, padding, EOS tokens, and polygon repair",
        "use_when": "Diffusion/flow over fixed padded token tensors",
    },
    {
        "representation": "Room boxes plus class labels",
        "pros": "Simpler fixed-size target; good baseline for room placement",
        "risks": "Loses non-rectangular room shape; vector refinement still needed",
        "use_when": "Fast baseline or first-stage layout proposal",
    },
    {
        "representation": "Raster semantic masks",
        "pros": "Fits standard image diffusion; easy conditioning on outline mask",
        "risks": "Submission is not pixels; needs polygonization, label cleanup, and topology repair",
        "use_when": "Prototype, ablation, or first-stage generator",
    },
])
display(decision_table)

eda_takeaways = pd.Series({
    "conditioning": "Use the 0.3 m buffer-union-buffer outline as the only fixed input condition.",
    "target": "Train on entity_type == area rows grouped by plan_id; output typed room polygons.",
    "normalization": "Normalize coordinates per outline bounds or centroid/scale because plans vary strongly in extent and area.",
    "imbalance": "Room labels are imbalanced; monitor rare labels and density/coverage to avoid mode collapse.",
    "fixed_tensor_options": "Pad room/corner sequences or train a raster semantic-mask baseline with vectorization afterward.",
    "evaluation_alignment": "FID and density/coverage reward realism and diversity, so keep stochastic sampling with seed 42 and inspect distribution coverage, not only reconstruction quality.",
})
display(eda_takeaways.to_frame("decision implication"))

import os
import random


def seed_everything(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        # Avoid CUDA probing here; this notebook currently runs on CPU in this environment.
    except ModuleNotFoundError:
        pass
    return seed

seed_everything(42)

plan_meta = (
    raw_df.loc[raw_df["entity_type"].eq("area"), ["plan_id", "building_id", "site_id", "apartment_id"]]
    .drop_duplicates("plan_id")
    .set_index("plan_id")
    .sort_index()
)
plan_ids = plan_meta.index.to_numpy(dtype=int)

site_ids = np.array(sorted(plan_meta["site_id"].unique()))
split_rng = np.random.default_rng(SEED)
shuffled_site_ids = site_ids.copy()
split_rng.shuffle(shuffled_site_ids)

n_sites = len(shuffled_site_ids)
n_train_sites = int(0.80 * n_sites)
n_val_sites = int(0.10 * n_sites)

train_site_ids = shuffled_site_ids[:n_train_sites]
val_site_ids = shuffled_site_ids[n_train_sites:n_train_sites + n_val_sites]
test_site_ids = shuffled_site_ids[n_train_sites + n_val_sites:]

site_to_split = {
    **{int(site_id): "train" for site_id in train_site_ids},
    **{int(site_id): "val" for site_id in val_site_ids},
    **{int(site_id): "test" for site_id in test_site_ids},
}
plan_meta["split"] = plan_meta["site_id"].map(site_to_split)

train_plan_ids = plan_meta.index[plan_meta["split"].eq("train")].to_numpy(dtype=int)
val_plan_ids = plan_meta.index[plan_meta["split"].eq("val")].to_numpy(dtype=int)
test_plan_ids = plan_meta.index[plan_meta["split"].eq("test")].to_numpy(dtype=int)

train_buildings = set(plan_meta.loc[train_plan_ids, "building_id"])
train_sites = set(plan_meta.loc[train_plan_ids, "site_id"])

split_summary = pd.DataFrame([
    {
        "split": "train",
        "plans": len(train_plan_ids),
        "sites": len(train_site_ids),
        "buildings": plan_meta.loc[train_plan_ids, "building_id"].nunique(),
        "share_building_with_train": 1.0,
        "share_site_with_train": 1.0,
    },
    {
        "split": "val",
        "plans": len(val_plan_ids),
        "sites": len(val_site_ids),
        "buildings": plan_meta.loc[val_plan_ids, "building_id"].nunique(),
        "share_building_with_train": plan_meta.loc[val_plan_ids, "building_id"].isin(train_buildings).mean(),
        "share_site_with_train": plan_meta.loc[val_plan_ids, "site_id"].isin(train_sites).mean(),
    },
    {
        "split": "test",
        "plans": len(test_plan_ids),
        "sites": len(test_site_ids),
        "buildings": plan_meta.loc[test_plan_ids, "building_id"].nunique(),
        "share_building_with_train": plan_meta.loc[test_plan_ids, "building_id"].isin(train_buildings).mean(),
        "share_site_with_train": plan_meta.loc[test_plan_ids, "site_id"].isin(train_sites).mean(),
    },
]).set_index("split")

split_summary[["share_building_with_train", "share_site_with_train"]] = (
    split_summary[["share_building_with_train", "share_site_with_train"]].round(3)
)
display(split_summary)

GRID_SIZE = 256
COORD_MAX = GRID_SIZE - 1
N_COORD_BITS = 8


def bounds_with_margin(bounds, margin_ratio=0.02, min_margin_m=0.50):
    minx, miny, maxx, maxy = bounds
    width = max(maxx - minx, 1e-6)
    height = max(maxy - miny, 1e-6)
    margin = max(width, height) * margin_ratio
    margin = max(margin, min_margin_m)
    return (minx - margin, miny - margin, maxx + margin, maxy + margin)


def coords_to_grid(coords_xy, bounds):
    coords_xy = np.asarray(coords_xy, dtype=np.float32)
    minx, miny, maxx, maxy = bounds
    scale = np.array([maxx - minx, maxy - miny], dtype=np.float32)
    offset = np.array([minx, miny], dtype=np.float32)
    normalized = (coords_xy - offset) / np.maximum(scale, 1e-6)
    grid = np.rint(np.clip(normalized, 0.0, 1.0) * COORD_MAX).astype(np.int64)
    return grid


def grid_to_coords(grid_xy, bounds):
    grid_xy = np.asarray(grid_xy, dtype=np.float32)
    minx, miny, maxx, maxy = bounds
    scale = np.array([maxx - minx, maxy - miny], dtype=np.float32)
    offset = np.array([minx, miny], dtype=np.float32)
    return offset + (grid_xy / COORD_MAX) * scale


def grid_to_continuous(grid_xy):
    return (np.asarray(grid_xy, dtype=np.float32) / COORD_MAX) * 2.0 - 1.0


def continuous_to_grid(values):
    values = np.asarray(values, dtype=np.float32)
    return np.rint(np.clip((values + 1.0) / 2.0, 0.0, 1.0) * COORD_MAX).astype(np.int64)


def int_to_bits(values, n_bits=N_COORD_BITS):
    values = np.asarray(values, dtype=np.int64)
    shifts = np.arange(n_bits - 1, -1, -1, dtype=np.int64)
    return ((values[..., None] >> shifts) & 1).astype(np.float32)


def bits_to_int(bits):
    bits = np.asarray(bits).astype(np.int64)
    shifts = np.arange(bits.shape[-1] - 1, -1, -1, dtype=np.int64)
    return (bits * (1 << shifts)).sum(axis=-1).astype(np.int64)

roundtrip_values = np.array([0, 1, 2, 127, 128, 254, 255])
assert np.array_equal(bits_to_int(int_to_bits(roundtrip_values)), roundtrip_values)
assert np.array_equal(continuous_to_grid(grid_to_continuous(roundtrip_values)), roundtrip_values)
print("quantization and bit roundtrips passed")

corner_hist = (
    rooms_gdf.groupby(["roomtype", "n_exterior_vertices"])
    .size()
    .rename("count")
    .reset_index()
    .sort_values(["roomtype", "n_exterior_vertices"])
)

corner_summary = rooms_gdf.groupby("roomtype")["n_exterior_vertices"].describe(
    percentiles=[.50, .75, .90, .95, .99]
).round(2)

display(corner_summary)
display(corner_hist.head(40))

fig, ax = plt.subplots(figsize=(12, 4))
for roomtype, group in corner_hist.groupby("roomtype"):
    ax.plot(group["n_exterior_vertices"], group["count"], marker="o", linewidth=1, label=roomtype)
ax.set_yscale("log")
ax.set_xlabel("exterior vertices, including closing coordinate")
ax.set_ylabel("count, log scale")
ax.set_title("Corner-count histograms by room type")
ax.legend(ncol=3, fontsize=8)
plt.tight_layout()
plt.savefig("corner_histograms.png")
plt.close(fig)

ROOMTYPE_TO_ID = {label: i + 1 for i, label in enumerate(sorted(rooms_gdf["roomtype"].unique()))}
ID_TO_ROOMTYPE = {v: k for k, v in ROOMTYPE_TO_ID.items()}
PAD_ROOMTYPE_ID = 0


def polygon_exterior_coords(geom):
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda part: part.area)
    coords = np.asarray(geom.exterior.coords, dtype=np.float32)
    if len(coords) > 1 and np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]
    return coords


def canonicalize_loop(grid_xy):
    grid_xy = np.asarray(grid_xy, dtype=np.int64)
    if len(grid_xy) == 0:
        return grid_xy
    order = np.lexsort((grid_xy[:, 1], grid_xy[:, 0]))
    start = int(order[0])
    return np.concatenate([grid_xy[start:], grid_xy[:start]], axis=0)


def plan_to_vector_tokens(plan_id, max_corners=None):
    plan_rooms = get_plan_rooms(plan_id).copy().sort_values(["roomtype", "area_m2"], ascending=[True, False])
    outline = get_plan_outline(plan_id)
    bounds = bounds_with_margin(outline.bounds)
    rows = []
    room_records = []

    for room_index, (_, room) in enumerate(plan_rooms.iterrows()):
        coords = polygon_exterior_coords(room.geometry)
        grid = canonicalize_loop(coords_to_grid(coords, bounds))
        if max_corners is not None:
            grid = grid[:max_corners]
        cont = grid_to_continuous(grid)
        roomtype_id = ROOMTYPE_TO_ID[room.roomtype]
        room_records.append({
            "plan_id": plan_id,
            "room_index": room_index,
            "roomtype": room.roomtype,
            "roomtype_id": roomtype_id,
            "n_corners": len(grid),
            "area_m2": room.area_m2,
        })
        for corner_index, ((gx, gy), (cx, cy)) in enumerate(zip(grid, cont)):
            rows.append({
                "plan_id": plan_id,
                "room_index": room_index,
                "corner_index": corner_index,
                "roomtype": room.roomtype,
                "roomtype_id": roomtype_id,
                "x_grid": int(gx),
                "y_grid": int(gy),
                "x_cont": float(cx),
                "y_cont": float(cy),
            })

    token_df = pd.DataFrame(rows)
    room_df = pd.DataFrame(room_records)
    return token_df, room_df, bounds

sample_token_df, sample_room_df, sample_vector_bounds = plan_to_vector_tokens(sample_plan_id)
display(sample_room_df.head())
display(sample_token_df.head(20))

def plot_quantized_plan(plan_id):
    token_df, room_df, _ = plan_to_vector_tokens(plan_id)
    fig, ax = plt.subplots(figsize=(6, 6))
    for room_index, group in token_df.groupby("room_index"):
        xy = group[["x_grid", "y_grid"]].to_numpy()
        if len(xy) == 0:
            continue
        closed = np.vstack([xy, xy[0]])
        label = group["roomtype"].iloc[0]
        ax.plot(closed[:, 0], closed[:, 1], marker="o", linewidth=1, markersize=2, label=label)
    ax.set_title(f"Quantized polygon loops, plan_id={plan_id}")
    ax.set_xlim(0, COORD_MAX)
    ax.set_ylim(0, COORD_MAX)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=7, ncol=2, loc="upper right")
    plt.tight_layout()
    plt.savefig(f"quantized_plan_{plan_id}.png")
    plt.close(fig)

plot_quantized_plan(sample_plan_id)

MAX_ROOMS = int(np.ceil(plan_stats.loc[train_plan_ids, "room_count"].quantile(0.95)))
MAX_CORNERS = int(np.ceil(rooms_gdf.loc[rooms_gdf["plan_id"].isin(train_plan_ids), "n_exterior_vertices"].quantile(0.95)))
MAX_CORNERS = max(MAX_CORNERS - 1, 4)  # remove closing coordinate, keep at least rectangle capacity

print({"MAX_ROOMS": MAX_ROOMS, "MAX_CORNERS": MAX_CORNERS})


def plan_to_padded_arrays(plan_id, max_rooms=MAX_ROOMS, max_corners=MAX_CORNERS):
    token_df, room_df, bounds = plan_to_vector_tokens(plan_id, max_corners=max_corners)
    coords = np.zeros((max_rooms, max_corners, 2), dtype=np.float32)
    grid = np.zeros((max_rooms, max_corners, 2), dtype=np.int64)
    bits = np.zeros((max_rooms, max_corners, 2, N_COORD_BITS), dtype=np.float32)
    corner_mask = np.zeros((max_rooms, max_corners), dtype=bool)
    room_mask = np.zeros((max_rooms,), dtype=bool)
    roomtype_ids = np.zeros((max_rooms,), dtype=np.int64)

    for room_index, room_tokens in token_df.groupby("room_index"):
        if room_index >= max_rooms:
            continue
        n = min(len(room_tokens), max_corners)
        xy_grid = room_tokens[["x_grid", "y_grid"]].to_numpy(dtype=np.int64)[:n]
        xy_cont = room_tokens[["x_cont", "y_cont"]].to_numpy(dtype=np.float32)[:n]
        coords[room_index, :n] = xy_cont
        grid[room_index, :n] = xy_grid
        bits[room_index, :n] = int_to_bits(xy_grid)
        corner_mask[room_index, :n] = True
        room_mask[room_index] = True
        roomtype_ids[room_index] = int(room_tokens["roomtype_id"].iloc[0])

    outline = get_plan_outline(plan_id)
    outline_grid = canonicalize_loop(coords_to_grid(polygon_exterior_coords(outline), bounds))
    outline_cont = grid_to_continuous(outline_grid)

    return {
        "coords": coords,
        "grid": grid,
        "bits": bits,
        "corner_mask": corner_mask,
        "room_mask": room_mask,
        "roomtype_ids": roomtype_ids,
        "outline_cont": outline_cont.astype(np.float32),
        "bounds": np.asarray(bounds, dtype=np.float32),
        "plan_id": int(plan_id),
    }

sample_arrays = plan_to_padded_arrays(sample_plan_id)
for key, value in sample_arrays.items():
    if hasattr(value, "shape"):
        print(key, value.shape, value.dtype)
    else:
        print(key, type(value).__name__)

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None
    Dataset = object


MAX_OUTLINE_CORNERS = 192
DIFFUSION_STEPS = 1000


def pad_outline_tokens(outline_cont, max_outline_corners=MAX_OUTLINE_CORNERS):
    outline_cont = np.asarray(outline_cont, dtype=np.float32)
    outline_tokens = np.zeros((max_outline_corners, 2), dtype=np.float32)
    outline_mask = np.zeros((max_outline_corners,), dtype=bool)
    n = min(len(outline_cont), max_outline_corners)
    if n:
        outline_tokens[:n] = outline_cont[:n]
        outline_mask[:n] = True
    return outline_tokens, outline_mask


if TORCH_AVAILABLE:
    class MSDVectorFloorplanDataset(Dataset):
        def __init__(self, plan_ids, max_rooms=MAX_ROOMS, max_corners=MAX_CORNERS, max_outline_corners=MAX_OUTLINE_CORNERS):
            self.plan_ids = [int(pid) for pid in plan_ids]
            self.max_rooms = max_rooms
            self.max_corners = max_corners
            self.max_outline_corners = max_outline_corners

        def __len__(self):
            return len(self.plan_ids)

        def __getitem__(self, index):
            item = plan_to_padded_arrays(self.plan_ids[index], self.max_rooms, self.max_corners)
            outline_tokens, outline_mask = pad_outline_tokens(item["outline_cont"], self.max_outline_corners)
            return {
                "coords": torch.from_numpy(item["coords"]),
                "grid": torch.from_numpy(item["grid"]),
                "bits": torch.from_numpy(item["bits"]),
                "corner_mask": torch.from_numpy(item["corner_mask"]),
                "room_mask": torch.from_numpy(item["room_mask"]),
                "roomtype_ids": torch.from_numpy(item["roomtype_ids"]),
                "outline_cont": torch.from_numpy(outline_tokens),
                "outline_mask": torch.from_numpy(outline_mask),
                "plan_id": torch.tensor(item["plan_id"], dtype=torch.long),
            }


    class OutlineConditionedVectorDenoiser(nn.Module):
        def __init__(self, n_roomtypes, max_rooms=MAX_ROOMS, max_corners=MAX_CORNERS, hidden_dim=128, n_layers=4, n_heads=4):
            super().__init__()
            self.n_roomtypes = n_roomtypes
            self.max_rooms = max_rooms
            self.max_corners = max_corners
            self.coord_proj = nn.Linear(2, hidden_dim)
            self.outline_proj = nn.Linear(2, hidden_dim)
            self.room_emb = nn.Embedding(n_roomtypes + 1, hidden_dim)
            self.room_pos_emb = nn.Embedding(max_rooms, hidden_dim)
            self.corner_pos_emb = nn.Embedding(max_corners, hidden_dim)
            self.time_proj = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
            self.condition_norm = nn.LayerNorm(hidden_dim)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.noise_head = nn.Linear(hidden_dim, 2)
            self.bit_head = nn.Linear(hidden_dim, 2 * N_COORD_BITS)
            self.roomtype_head = nn.Linear(hidden_dim, n_roomtypes + 1)

        def encode_outline(self, outline_cont, outline_mask):
            outline_x = self.outline_proj(outline_cont)
            mask = outline_mask.float().unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            pooled = (outline_x * mask).sum(dim=1) / denom
            return self.condition_norm(pooled)

        def forward(self, noisy_coords, roomtype_ids, corner_mask, room_mask, outline_cont, outline_mask, t):
            batch, rooms, corners, _ = noisy_coords.shape
            tokens = noisy_coords.reshape(batch, rooms * corners, 2)
            valid = corner_mask.reshape(batch, rooms * corners)

            room_positions = torch.arange(rooms, device=noisy_coords.device)
            corner_positions = torch.arange(corners, device=noisy_coords.device)
            room_pos = self.room_pos_emb(room_positions)[None, :, None, :].expand(batch, rooms, corners, -1)
            corner_pos = self.corner_pos_emb(corner_positions)[None, None, :, :].expand(batch, rooms, corners, -1)
            room_ids = roomtype_ids[:, :, None].expand(batch, rooms, corners).reshape(batch, rooms * corners)

            time = t.float().view(batch, 1) / float(DIFFUSION_STEPS)
            condition = self.encode_outline(outline_cont, outline_mask)
            condition_tokens = condition[:, None, :].expand(batch, rooms * corners, -1)

            x = (
                self.coord_proj(tokens)
                + self.room_emb(room_ids)
                + room_pos.reshape(batch, rooms * corners, -1)
                + corner_pos.reshape(batch, rooms * corners, -1)
                + self.time_proj(time)[:, None, :]
                + condition_tokens
            )
            x = self.encoder(x, src_key_padding_mask=~valid)
            pred_noise = self.noise_head(x).reshape(batch, rooms, corners, 2)
            pred_bits = self.bit_head(x).reshape(batch, rooms, corners, 2, N_COORD_BITS)

            room_features = x.reshape(batch, rooms, corners, -1)
            corner_weights = corner_mask.float().unsqueeze(-1)
            room_features = (room_features * corner_weights).sum(dim=2) / corner_weights.sum(dim=2).clamp_min(1.0)
            roomtype_logits = self.roomtype_head(room_features)
            return pred_noise, pred_bits, roomtype_logits


    def cosine_alpha_bar(t, total_steps=DIFFUSION_STEPS, s=0.008):
        t = t.float() / float(total_steps)
        return torch.cos(((t + s) / (1 + s)) * torch.pi * 0.5).pow(2)


    def add_diffusion_noise(coords, corner_mask, t, generator=None):
        noise = torch.randn(coords.shape, device=coords.device, dtype=coords.dtype, generator=generator)
        alpha_bar = cosine_alpha_bar(t).view(-1, 1, 1, 1).clamp(1e-5, 0.999)
        noisy_coords = alpha_bar.sqrt() * coords + (1.0 - alpha_bar).sqrt() * noise
        noisy_coords = torch.where(corner_mask[..., None], noisy_coords, coords)
        return noisy_coords, noise


    def diffusion_training_step(model, batch, generator=None, bit_loss_weight=0.02, roomtype_loss_weight=0.25, roomtype_condition_dropout=0.35):
        coords = batch["coords"]
        corner_mask = batch["corner_mask"]
        room_mask = batch["room_mask"]
        roomtype_targets = batch["roomtype_ids"]
        batch_size = coords.shape[0]
        t = torch.randint(1, DIFFUSION_STEPS, (batch_size,), device=coords.device, generator=generator)
        noisy_coords, noise = add_diffusion_noise(coords, corner_mask, t, generator=generator)

        roomtype_inputs = roomtype_targets.clone()
        if roomtype_condition_dropout > 0:
            drop = torch.rand(roomtype_inputs.shape, device=roomtype_inputs.device, generator=generator) < roomtype_condition_dropout
            roomtype_inputs = roomtype_inputs.masked_fill(drop & room_mask, PAD_ROOMTYPE_ID)

        pred_noise, pred_bits, roomtype_logits = model(
            noisy_coords,
            roomtype_inputs,
            corner_mask,
            room_mask,
            batch["outline_cont"],
            batch["outline_mask"],
            t,
        )
        noise_loss = F.mse_loss(pred_noise[corner_mask], noise[corner_mask])
        bit_loss = F.binary_cross_entropy_with_logits(pred_bits[corner_mask], batch["bits"][corner_mask])
        roomtype_loss = F.cross_entropy(roomtype_logits[room_mask], roomtype_targets[room_mask])
        loss = noise_loss + bit_loss_weight * bit_loss + roomtype_loss_weight * roomtype_loss
        return {
            "loss": loss,
            "noise_loss": noise_loss.detach(),
            "bit_loss": bit_loss.detach(),
            "roomtype_loss": roomtype_loss.detach(),
        }


    seed_everything(SEED)
    train_dataset = MSDVectorFloorplanDataset(train_plan_ids[:128])
    loader_generator = torch.Generator().manual_seed(SEED)
    example_batch = next(iter(DataLoader(train_dataset, batch_size=4, shuffle=True, generator=loader_generator)))
    model = OutlineConditionedVectorDenoiser(n_roomtypes=len(ROOMTYPE_TO_ID))
    step_generator = torch.Generator().manual_seed(SEED)
    losses = diffusion_training_step(model, example_batch, generator=step_generator)
    print({key: round(float(value.detach()), 4) for key, value in losses.items()})
else:
    print("PyTorch is not installed. The outline-conditioned dataset/model code is ready but was not executed.")

from shapely.affinity import scale as shapely_scale, translate as shapely_translate
from matplotlib.patches import Patch


ROOMTYPE_COLORS = {
    roomtype: plt.get_cmap("tab20")(idx % 20)
    for idx, roomtype in enumerate(sorted(rooms_gdf["roomtype"].dropna().unique()))
}
ROOMTYPE_ORDER = sorted(ROOMTYPE_TO_ID)


def outline_features_for_plan(plan_id):
    outline = get_plan_outline(plan_id)
    minx, miny, maxx, maxy = outline.bounds
    width = max(maxx - minx, 1e-6)
    height = max(maxy - miny, 1e-6)
    area = outline.area
    perimeter = outline.length
    return pd.Series({
        "outline_area_m2": area,
        "outline_perimeter_m": perimeter,
        "outline_width_m": width,
        "outline_height_m": height,
        "outline_aspect": width / height,
        "outline_compactness": (4 * np.pi * area) / max(perimeter ** 2, 1e-6),
    })


outline_feature_df = pd.DataFrame({
    int(plan_id): outline_features_for_plan(plan_id)
    for plan_id in plan_ids
}).T
outline_feature_df.index.name = "plan_id"

feature_cols = [
    "outline_area_m2",
    "outline_perimeter_m",
    "outline_width_m",
    "outline_height_m",
    "outline_aspect",
    "outline_compactness",
]
train_outline_features = outline_feature_df.loc[train_plan_ids, feature_cols]
feature_center = train_outline_features.mean()
feature_scale = train_outline_features.std().replace(0, 1)

plan_roomtype_counts = (
    rooms_gdf.pivot_table(index="plan_id", columns="roomtype", values="geometry", aggfunc="size", fill_value=0)
    .reindex(index=plan_ids, columns=ROOMTYPE_ORDER, fill_value=0)
    .astype(float)
)
plan_program_df = plan_stats[["room_count", "total_area_m2"]].join(plan_roomtype_counts)
plan_program_df["target_coverage_prior"] = (
    plan_stats["total_area_m2"] / outline_feature_df["outline_area_m2"]
).clip(0.55, 1.0)

train_room_area_stats = (
    rooms_gdf.loc[rooms_gdf["plan_id"].isin(train_plan_ids)]
    .assign(log_area=lambda df: np.log1p(df["area_m2"].clip(lower=0.01)))
    .groupby("roomtype")["log_area"]
    .agg(
        median="median",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
    )
)
train_room_area_stats["iqr"] = (train_room_area_stats["q75"] - train_room_area_stats["q25"]).replace(0, 0.5)


def polygon_compactness(geom):
    if geom.is_empty or geom.length <= 0:
        return 0.0
    return float((4 * np.pi * geom.area) / max(geom.length ** 2, 1e-6))


def room_naturalness_penalty(prediction):
    if len(prediction) == 0:
        return 3.0
    penalties = []
    for _, row in prediction.iterrows():
        stats = train_room_area_stats.loc[row.roomtype] if row.roomtype in train_room_area_stats.index else None
        log_area = np.log1p(max(float(row.area_m2), 0.01))
        if stats is not None:
            area_penalty = min(abs(log_area - float(stats["median"])) / max(float(stats["iqr"]), 0.25), 3.0)
        else:
            area_penalty = 1.0
        compactness = polygon_compactness(row.geometry)
        compactness_penalty = max(0.0, 0.16 - compactness) / 0.16
        tiny_penalty = 1.0 if row.area_m2 < 0.5 else 0.0
        penalties.append(0.55 * area_penalty + 0.35 * compactness_penalty + 0.10 * tiny_penalty)
    return float(np.mean(penalties))


def normalized_outline_features(plan_id_values):
    return ((outline_feature_df.loc[plan_id_values, feature_cols] - feature_center) / feature_scale).to_numpy(dtype=float)


def ranked_outline_matches(query_plan_id, candidate_plan_ids=train_plan_ids, same_site_ok=False, same_building_ok=False, top_k=128):
    query_plan_id = int(query_plan_id)
    query_meta = plan_meta.loc[query_plan_id]
    candidate_plan_ids = np.asarray(candidate_plan_ids, dtype=int)
    candidate_meta = plan_meta.loc[candidate_plan_ids]

    keep = np.ones(len(candidate_plan_ids), dtype=bool)
    if not same_site_ok:
        keep &= candidate_meta["site_id"].to_numpy() != int(query_meta["site_id"])
    if not same_building_ok:
        keep &= candidate_meta["building_id"].to_numpy() != int(query_meta["building_id"])

    filtered_plan_ids = candidate_plan_ids[keep]
    if len(filtered_plan_ids) == 0:
        filtered_plan_ids = candidate_plan_ids.copy()

    query = normalized_outline_features([query_plan_id])[0]
    candidates = normalized_outline_features(filtered_plan_ids)
    distances = np.linalg.norm(candidates - query[None, :], axis=1)

    ranked = pd.DataFrame({
        "source_plan_id": filtered_plan_ids,
        "outline_distance": distances,
    }).sort_values(["outline_distance", "source_plan_id"], ignore_index=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked.head(top_k).copy()


def weighted_program_prior(query_plan_id, candidate_plan_ids=train_plan_ids, program_k=64, temperature=0.35):
    matches = ranked_outline_matches(query_plan_id, candidate_plan_ids=candidate_plan_ids, top_k=program_k)
    distances = matches["outline_distance"].to_numpy(dtype=float)
    scaled = -distances / max(float(temperature), 1e-6)
    scaled -= scaled.max()
    weights = np.exp(scaled)
    weights /= weights.sum()

    neighbor_ids = matches["source_plan_id"].to_numpy(dtype=int)
    neighbor_programs = plan_program_df.loc[neighbor_ids]
    prior_values = weights @ neighbor_programs.to_numpy(dtype=float)
    prior = pd.Series(prior_values, index=neighbor_programs.columns)
    prior["program_entropy"] = float(-(weights * np.log(weights + 1e-12)).sum())
    return prior, matches.assign(program_weight=weights)


def fit_rooms_to_outline(source_plan_id, target_plan_id):
    source_rooms = get_plan_rooms(source_plan_id).copy()
    source_outline = get_plan_outline(source_plan_id)
    target_outline = get_plan_outline(target_plan_id)

    sx0, sy0, sx1, sy1 = source_outline.bounds
    tx0, ty0, tx1, ty1 = target_outline.bounds
    source_width = max(sx1 - sx0, 1e-6)
    source_height = max(sy1 - sy0, 1e-6)
    target_width = max(tx1 - tx0, 1e-6)
    target_height = max(ty1 - ty0, 1e-6)

    xfact = target_width / source_width
    yfact = target_height / source_height
    fitted_geometries = []
    for geom in source_rooms.geometry:
        moved_to_origin = shapely_translate(geom, xoff=-sx0, yoff=-sy0)
        resized = shapely_scale(moved_to_origin, xfact=xfact, yfact=yfact, origin=(0, 0))
        moved_to_target = shapely_translate(resized, xoff=tx0, yoff=ty0)
        clipped = moved_to_target.intersection(target_outline)
        if not clipped.is_empty:
            clipped = clipped.buffer(0)
        fitted_geometries.append(clipped)

    prediction = source_rooms[["roomtype"]].copy()
    prediction["source_plan_id"] = int(source_plan_id)
    prediction["target_plan_id"] = int(target_plan_id)
    prediction = gpd.GeoDataFrame(prediction, geometry=fitted_geometries)
    prediction = prediction.loc[~prediction.geometry.is_empty].copy()
    prediction["area_m2"] = prediction.geometry.area
    prediction = prediction.loc[prediction["area_m2"] > 0.05].copy()
    return prediction


def roomtype_count_vector(room_gdf):
    if len(room_gdf) == 0:
        return pd.Series(0, index=ROOMTYPE_ORDER, dtype=float)
    return room_gdf["roomtype"].value_counts().reindex(ROOMTYPE_ORDER, fill_value=0).astype(float)


def score_fitted_prediction(plan_id, prediction, program_prior, outline_distance):
    target_outline = get_plan_outline(plan_id)
    pred_union = union_geometries(prediction.geometry) if len(prediction) else Polygon()
    coverage = pred_union.intersection(target_outline).area / target_outline.area if target_outline.area else 0.0

    expected_room_count = max(float(program_prior["room_count"]), 1.0)
    room_count_error = abs(len(prediction) - expected_room_count) / expected_room_count

    expected_counts = program_prior[ROOMTYPE_ORDER].astype(float)
    predicted_counts = roomtype_count_vector(prediction)
    expected_total = max(float(expected_counts.sum()), 1.0)
    roomtype_count_error = float(np.abs(predicted_counts - expected_counts).sum() / expected_total)

    expected_coverage = float(program_prior["target_coverage_prior"])
    coverage_error = abs(coverage - expected_coverage)
    naturalness_penalty = room_naturalness_penalty(prediction)

    score = (
        0.55 * float(outline_distance)
        + 1.25 * room_count_error
        + 0.85 * roomtype_count_error
        + 2.00 * coverage_error
        + 0.55 * naturalness_penalty
    )
    return pd.Series({
        "candidate_score": score,
        "program_room_count": expected_room_count,
        "predicted_coverage": coverage,
        "expected_coverage": expected_coverage,
        "program_room_count_error": room_count_error,
        "program_roomtype_count_error": roomtype_count_error,
        "program_coverage_error": coverage_error,
        "naturalness_penalty": naturalness_penalty,
    })


def ranked_layout_candidates(plan_id, candidate_plan_ids=train_plan_ids, top_k=192, program_k=64, same_site_ok=False, same_building_ok=False):
    program_prior, program_neighbors = weighted_program_prior(
        plan_id,
        candidate_plan_ids=candidate_plan_ids,
        program_k=program_k,
    )
    outline_matches = ranked_outline_matches(
        plan_id,
        candidate_plan_ids=candidate_plan_ids,
        top_k=top_k,
        same_site_ok=same_site_ok,
        same_building_ok=same_building_ok,
    )

    scored_rows = []
    for _, match in outline_matches.iterrows():
        prediction = fit_rooms_to_outline(int(match["source_plan_id"]), plan_id)
        score = score_fitted_prediction(plan_id, prediction, program_prior, match["outline_distance"])
        scored_rows.append({**match.to_dict(), **score.to_dict()})

    scored = pd.DataFrame(scored_rows).sort_values(["candidate_score", "outline_distance"], ignore_index=True)
    scored["candidate_rank"] = np.arange(1, len(scored) + 1)
    return scored, program_prior, program_neighbors


def sample_scored_candidates(scored_candidates, n_samples=4, temperature=0.08, rng=None, pool_multiplier=6, include_best=True):
    if rng is None:
        rng = np.random.default_rng(SEED)
    n_samples = int(min(max(n_samples, 1), len(scored_candidates)))
    pool_size = min(len(scored_candidates), max(n_samples * pool_multiplier, n_samples))
    pool = scored_candidates.head(pool_size).copy()

    if include_best and n_samples == 1:
        sampled = pool.head(1).copy().reset_index(drop=True)
        sampled["sampling_probability"] = 1.0
        return sampled

    fixed = pool.head(1).copy() if include_best else pool.head(0).copy()
    remaining = pool.iloc[1:].copy() if include_best else pool.copy()
    n_random = n_samples - len(fixed)

    scores = remaining["candidate_score"].to_numpy(dtype=float)
    scaled = -scores / max(float(temperature), 1e-6)
    scaled -= scaled.max()
    probabilities = np.exp(scaled)
    probabilities /= probabilities.sum()
    choice_indices = rng.choice(len(remaining), size=n_random, replace=False, p=probabilities)
    sampled_random = remaining.iloc[np.atleast_1d(choice_indices)].copy()
    sampled_random["sampling_probability"] = probabilities[np.atleast_1d(choice_indices)]
    fixed["sampling_probability"] = 1.0
    return pd.concat([fixed, sampled_random], ignore_index=True)


def attach_candidate_attrs(prediction, selected, sample_index, top_k, temperature):
    prediction.attrs.update({
        "source_plan_id": int(selected["source_plan_id"]),
        "outline_distance": float(selected["outline_distance"]),
        "match_rank": int(selected["rank"]),
        "candidate_rank": int(selected["candidate_rank"]),
        "candidate_score": float(selected["candidate_score"]),
        "sampling_probability": float(selected["sampling_probability"]),
        "sample_index": int(sample_index),
        "top_k": int(top_k),
        "temperature": float(temperature),
    })
    return prediction


def predict_rooms_program_aware_set(plan_id, n_samples=4, candidate_plan_ids=train_plan_ids, top_k=96, program_k=48, temperature=0.08, sample_seed=SEED, same_site_ok=False, same_building_ok=False):
    scored, program_prior, program_neighbors = ranked_layout_candidates(
        plan_id,
        candidate_plan_ids=candidate_plan_ids,
        top_k=top_k,
        program_k=program_k,
        same_site_ok=same_site_ok,
        same_building_ok=same_building_ok,
    )
    rng = np.random.default_rng(int(sample_seed) + int(plan_id) * 1000)
    sampled = sample_scored_candidates(scored, n_samples=n_samples, temperature=temperature, rng=rng)

    predictions = []
    for sample_index, (_, selected) in enumerate(sampled.iterrows()):
        prediction = fit_rooms_to_outline(int(selected["source_plan_id"]), plan_id)
        prediction = attach_candidate_attrs(prediction, selected, sample_index, top_k, temperature)
        predictions.append(prediction)
    return predictions, scored, program_prior, program_neighbors


# Backward-compatible helper name for older cells.
def predict_rooms_diverse_set(plan_id, n_samples=4, candidate_plan_ids=train_plan_ids, top_k=96, temperature=0.08, sample_seed=SEED, same_site_ok=False, same_building_ok=False):
    predictions, _, _, _ = predict_rooms_program_aware_set(
        plan_id,
        n_samples=n_samples,
        candidate_plan_ids=candidate_plan_ids,
        top_k=top_k,
        temperature=temperature,
        sample_seed=sample_seed,
        same_site_ok=same_site_ok,
        same_building_ok=same_building_ok,
    )
    return predictions


def plot_room_polygons(ax, room_gdf, outline=None, title=None, show_labels=True):
    if len(room_gdf):
        for roomtype, group in room_gdf.groupby("roomtype"):
            group.plot(
                ax=ax,
                facecolor=ROOMTYPE_COLORS.get(roomtype, "#cccccc"),
                edgecolor="white",
                linewidth=0.9,
                alpha=0.88,
            )
    if outline is not None:
        gpd.GeoSeries([outline]).plot(ax=ax, facecolor="none", edgecolor="black", linewidth=2.0)
    if show_labels and len(room_gdf):
        for _, row in room_gdf.iterrows():
            point = row.geometry.representative_point()
            ax.text(point.x, point.y, str(row.roomtype)[:10], ha="center", va="center", fontsize=7)
    ax.set_title(title or "")
    ax.axis("equal")
    ax.axis("off")


def mean_best_same_type_iou(target_rooms, prediction):
    scores = []
    for _, target_row in target_rooms.iterrows():
        same_type = prediction.loc[prediction["roomtype"].eq(target_row.roomtype)]
        if same_type.empty:
            scores.append(0.0)
            continue
        best_iou = 0.0
        for pred_geom in same_type.geometry:
            union_area = target_row.geometry.union(pred_geom).area
            if union_area <= 0:
                continue
            inter_area = target_row.geometry.intersection(pred_geom).area
            best_iou = max(best_iou, inter_area / union_area)
        scores.append(best_iou)
    return float(np.mean(scores)) if scores else np.nan


def prediction_quality(plan_id, prediction):
    target_rooms = get_plan_rooms(plan_id)
    target_outline = get_plan_outline(plan_id)
    target_union = union_geometries(target_rooms.geometry)
    pred_union = union_geometries(prediction.geometry) if len(prediction) else Polygon()
    intersection_area = target_union.intersection(pred_union).area
    union_area = target_union.union(pred_union).area
    target_counts = roomtype_count_vector(target_rooms)
    pred_counts = roomtype_count_vector(prediction)

    return pd.Series({
        "plan_id": int(plan_id),
        "source_plan_id": int(prediction.attrs.get("source_plan_id", -1)),
        "match_rank": int(prediction.attrs.get("match_rank", -1)),
        "candidate_rank": int(prediction.attrs.get("candidate_rank", -1)),
        "candidate_score": float(prediction.attrs.get("candidate_score", np.nan)),
        "outline_distance": float(prediction.attrs.get("outline_distance", np.nan)),
        "sample_index": int(prediction.attrs.get("sample_index", -1)),
        "target_rooms": len(target_rooms),
        "predicted_rooms": len(prediction),
        "room_count_error": int(len(prediction) - len(target_rooms)),
        "roomtype_count_l1": int(np.abs(target_counts - pred_counts).sum()),
        "mean_best_same_type_iou": mean_best_same_type_iou(target_rooms, prediction),
        "target_area_m2": target_union.area,
        "predicted_area_m2": pred_union.area,
        "outline_area_m2": target_outline.area,
        "union_iou": intersection_area / union_area if union_area else np.nan,
        "prediction_outline_coverage": pred_union.intersection(target_outline).area / target_outline.area if target_outline.area else np.nan,
    })


def plot_prediction_gallery(plan_id, n_samples=4, top_k=96, program_k=48, temperature=0.08):
    target_rooms = get_plan_rooms(plan_id)
    outline = get_plan_outline(plan_id)
    predictions, scored, program_prior, program_neighbors = predict_rooms_program_aware_set(
        plan_id,
        n_samples=n_samples,
        top_k=top_k,
        program_k=program_k,
        temperature=temperature,
    )

    fig, axes = plt.subplots(1, 2 + len(predictions), figsize=(4 * (2 + len(predictions)), 5.5))
    plot_room_polygons(axes[0], gpd.GeoDataFrame({"roomtype": []}, geometry=[]), outline, f"Condition outline\nplan_id={plan_id}", show_labels=False)
    plot_room_polygons(axes[1], target_rooms, outline, f"Ground truth\n{len(target_rooms)} rooms")

    for ax, prediction in zip(axes[2:], predictions):
        plot_room_polygons(
            ax,
            prediction,
            outline,
            title=(
                f"Sample {prediction.attrs['sample_index']}\n"
                f"src={prediction.attrs['source_plan_id']} | score={prediction.attrs['candidate_score']:.2f}\n"
                f"rooms={len(prediction)} | cov={prediction_quality(plan_id, prediction)['prediction_outline_coverage']:.2f}"
            ),
        )

    present_roomtypes = sorted(set(target_rooms["roomtype"]).union(*(set(pred["roomtype"]) for pred in predictions)))
    handles = [Patch(facecolor=ROOMTYPE_COLORS[label], edgecolor="white", label=label) for label in present_roomtypes]
    fig.legend(handles=handles, loc="lower center", ncol=min(6, max(1, len(handles))), frameon=False)
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.savefig(f"prediction_gallery_{plan_id}.png")
    plt.close(fig)
    return predictions, scored, program_prior, program_neighbors


example_prediction_plan_ids = [int(pid) for pid in val_plan_ids[:3]]
quality_rows = []
for plan_id in example_prediction_plan_ids:
    predictions, scored, program_prior, _ = plot_prediction_gallery(plan_id, n_samples=4, top_k=96, program_k=48, temperature=0.08)
    print(f"plan_id={plan_id} expected rooms={program_prior['room_count']:.1f}, expected coverage={program_prior['target_coverage_prior']:.2f}")
    display(scored.head(8).round(3))
    for pred in predictions:
        quality_rows.append(prediction_quality(plan_id, pred))

prediction_quality_df = pd.DataFrame(quality_rows)
display(prediction_quality_df.round(3))

def predicted_arrays_to_gdf(coords_grid, roomtype_ids, corner_mask, bounds, min_area_m2=0.05):
    rows = []
    for room_index, roomtype_id in enumerate(np.asarray(roomtype_ids).astype(int)):
        if roomtype_id == PAD_ROOMTYPE_ID or roomtype_id not in ID_TO_ROOMTYPE:
            continue
        valid = np.asarray(corner_mask[room_index]).astype(bool)
        if valid.sum() < 3:
            continue
        grid_xy = np.asarray(coords_grid[room_index])[valid]
        xy = grid_to_coords(grid_xy, bounds)
        poly = Polygon(xy).buffer(0)
        if poly.is_empty or poly.area <= min_area_m2:
            continue
        rows.append({
            "room_index": room_index,
            "roomtype": ID_TO_ROOMTYPE[int(roomtype_id)],
            "area_m2": poly.area,
            "geometry": poly,
        })
    return gpd.GeoDataFrame(rows, geometry="geometry")


def plot_model_prediction_from_arrays(plan_id, coords_grid, roomtype_ids, corner_mask, bounds, title="Model prediction"):
    prediction_gdf = predicted_arrays_to_gdf(coords_grid, roomtype_ids, corner_mask, bounds)
    outline = get_plan_outline(plan_id)
    fig, ax = plt.subplots(figsize=(7, 7))
    plot_room_polygons(ax, prediction_gdf, outline, title)
    plt.tight_layout()
    plt.savefig(f"model_prediction_{plan_id}.png")
    plt.close(fig)
    return prediction_gdf


# Smoke test the conversion on the quantized ground-truth arrays.
quantized_ground_truth_gdf = predicted_arrays_to_gdf(
    sample_arrays["grid"],
    sample_arrays["roomtype_ids"],
    sample_arrays["corner_mask"],
    sample_arrays["bounds"],
)
print(f"converted {len(quantized_ground_truth_gdf)} polygons from padded arrays")

TRAINING_MAX_STEPS = 8
TRAINING_BATCH_SIZE = 2
TRAINING_SUBSET_SIZE = 192
RUN_DENOISER_TRAINING = False
ENABLE_NEURAL_REFINEMENT = False
CHECKPOINT_DIR = Path("artifacts")
CHECKPOINT_PATH = CHECKPOINT_DIR / "outline_conditioned_denoiser.pt"


def prediction_to_padded_arrays(prediction_gdf, plan_id, max_rooms=MAX_ROOMS, max_corners=MAX_CORNERS):
    outline = get_plan_outline(plan_id)
    bounds = bounds_with_margin(outline.bounds)

    coords = np.zeros((max_rooms, max_corners, 2), dtype=np.float32)
    grid = np.zeros((max_rooms, max_corners, 2), dtype=np.int64)
    bits = np.zeros((max_rooms, max_corners, 2, N_COORD_BITS), dtype=np.float32)
    corner_mask = np.zeros((max_rooms, max_corners), dtype=bool)
    room_mask = np.zeros((max_rooms,), dtype=bool)
    roomtype_ids = np.zeros((max_rooms,), dtype=np.int64)

    if len(prediction_gdf):
        ordered = prediction_gdf.copy().sort_values(["roomtype", "area_m2"], ascending=[True, False])
        for room_index, (_, room) in enumerate(ordered.iterrows()):
            if room_index >= max_rooms or room.roomtype not in ROOMTYPE_TO_ID:
                continue
            room_geom = room.geometry
            if room_geom.is_empty:
                continue
            room_grid = canonicalize_loop(coords_to_grid(polygon_exterior_coords(room_geom), bounds))[:max_corners]
            if len(room_grid) < 3:
                continue
            room_cont = grid_to_continuous(room_grid)
            n = len(room_grid)
            grid[room_index, :n] = room_grid
            coords[room_index, :n] = room_cont
            bits[room_index, :n] = int_to_bits(room_grid)
            corner_mask[room_index, :n] = True
            room_mask[room_index] = True
            roomtype_ids[room_index] = ROOMTYPE_TO_ID[room.roomtype]

    outline_grid = canonicalize_loop(coords_to_grid(polygon_exterior_coords(outline), bounds))
    outline_cont = grid_to_continuous(outline_grid)
    outline_tokens, outline_mask = pad_outline_tokens(outline_cont, MAX_OUTLINE_CORNERS)
    return {
        "coords": coords,
        "grid": grid,
        "bits": bits,
        "corner_mask": corner_mask,
        "room_mask": room_mask,
        "roomtype_ids": roomtype_ids,
        "outline_cont": outline_tokens.astype(np.float32),
        "outline_mask": outline_mask,
        "bounds": np.asarray(bounds, dtype=np.float32),
        "plan_id": int(plan_id),
    }


def clip_prediction_to_outline(prediction_gdf, plan_id, min_area_m2=0.05):
    if len(prediction_gdf) == 0:
        return prediction_gdf
    outline = get_plan_outline(plan_id)
    clipped = prediction_gdf.copy()
    clipped["geometry"] = clipped.geometry.map(lambda geom: geom.intersection(outline).buffer(0))
    clipped = clipped.loc[~clipped.geometry.is_empty].copy()
    clipped["area_m2"] = clipped.geometry.area
    return clipped.loc[clipped["area_m2"] > min_area_m2].copy()


def outline_denoiser_config(training_steps=None):
    return {
        "max_rooms": MAX_ROOMS,
        "max_corners": MAX_CORNERS,
        "max_outline_corners": MAX_OUTLINE_CORNERS,
        "n_roomtypes": len(ROOMTYPE_TO_ID),
        "grid_size": GRID_SIZE,
        "n_coord_bits": N_COORD_BITS,
        "training_steps": training_steps,
    }


def save_outline_denoiser_weights(model, checkpoint_path=CHECKPOINT_PATH, history=None, extra=None):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed; cannot save model weights.")
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": outline_denoiser_config(
            training_steps=len(history) if history is not None else None,
        ),
        "roomtype_to_id": ROOMTYPE_TO_ID,
        "id_to_roomtype": ID_TO_ROOMTYPE,
        "history": history or [],
        "extra": extra or {},
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"saved model weights to {checkpoint_path}")
    return checkpoint_path


def load_outline_denoiser_weights(model, checkpoint_path=CHECKPOINT_PATH, strict=True):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed; cannot load model weights.")
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=strict)
    return checkpoint


def load_outline_denoiser(checkpoint_path=CHECKPOINT_PATH, strict=True):
    if not TORCH_AVAILABLE:
        print("PyTorch is not installed; cannot load model weights.")
        return None
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path}")
        return None
    model = OutlineConditionedVectorDenoiser(n_roomtypes=len(ROOMTYPE_TO_ID))
    checkpoint = load_outline_denoiser_weights(model, checkpoint_path=checkpoint_path, strict=strict)
    model.eval()
    print(f"loaded model weights from {checkpoint_path}")
    return model


def train_outline_denoiser(max_steps=TRAINING_MAX_STEPS, batch_size=TRAINING_BATCH_SIZE, train_subset_size=TRAINING_SUBSET_SIZE, lr=2e-4, checkpoint_path=CHECKPOINT_PATH):
    if not TORCH_AVAILABLE:
        print("PyTorch is not installed; skipping training.")
        return None, pd.DataFrame()

    seed_everything(SEED)
    train_ids = train_plan_ids[:train_subset_size]
    dataset = MSDVectorFloorplanDataset(train_ids)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    model = OutlineConditionedVectorDenoiser(n_roomtypes=len(ROOMTYPE_TO_ID))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    step_generator = torch.Generator().manual_seed(SEED)

    history = []
    model.train()
    step = 0
    while step < max_steps:
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            losses = diffusion_training_step(model, batch, generator=step_generator)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            record = {key: float(value.detach()) for key, value in losses.items()}
            record["step"] = step + 1
            history.append(record)
            step += 1
            if step >= max_steps:
                break

    history_df = pd.DataFrame(history)
    save_outline_denoiser_weights(
        model,
        checkpoint_path=checkpoint_path,
        history=history,
        extra={
            "batch_size": batch_size,
            "train_subset_size": train_subset_size,
            "learning_rate": lr,
        },
    )
    display(history_df.tail())
    return model, history_df


def outline_coverage(plan_id, prediction_gdf):
    outline = get_plan_outline(plan_id)
    if len(prediction_gdf) == 0 or outline.area <= 0:
        return 0.0
    union = union_geometries(prediction_gdf.geometry)
    return float(union.intersection(outline).area / outline.area)


def geometry_polygon_parts(geom):
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type == "MultiPolygon":
        return list(geom.geoms)
    if geom.geom_type == "GeometryCollection":
        return [part for part in geom.geoms if part.geom_type == "Polygon" and not part.is_empty]
    return []


def best_gap_owner_index(repaired_gdf, gap):
    best_index = None
    best_score = -np.inf
    for index, row in repaired_gdf.iterrows():
        shared_boundary = row.geometry.boundary.intersection(gap.boundary).length
        distance = row.geometry.distance(gap)
        score = shared_boundary - 0.05 * distance
        if score > best_score:
            best_score = score
            best_index = index
    return best_index


def resolve_room_overlaps(prediction_gdf, min_area_m2=0.05):
    cleaned = prediction_gdf.copy().reset_index(drop=True)
    if len(cleaned) <= 1:
        return cleaned

    overlap_removed_area = 0.0
    order = cleaned.sort_values("area_m2", ascending=False).index.to_list()
    assigned = Polygon()
    new_geometries = [None] * len(cleaned)

    for index in order:
        geom = cleaned.at[index, "geometry"].buffer(0)
        if not assigned.is_empty:
            overlap = geom.intersection(assigned).area
            if overlap > 0:
                overlap_removed_area += overlap
                geom = geom.difference(assigned).buffer(0)
        new_geometries[index] = geom
        if not geom.is_empty:
            assigned = unary_union([assigned, geom]).buffer(0)

    cleaned["geometry"] = new_geometries
    cleaned = cleaned.loc[~cleaned.geometry.is_empty].copy()
    cleaned["area_m2"] = cleaned.geometry.area
    cleaned = cleaned.loc[cleaned["area_m2"] > min_area_m2].copy()
    cleaned.attrs.update(prediction_gdf.attrs)
    cleaned.attrs["overlap_removed_area_m2"] = float(overlap_removed_area)
    return gpd.GeoDataFrame(cleaned, geometry="geometry")


def repair_outline_coverage(
    prediction_gdf,
    plan_id,
    small_gap_area_m2=1.25,
    filler_min_area_m2=0.15,
    max_added_area_ratio=0.22,
    filler_roomtype="Corridor",
    allow_filler_rooms=False,
    max_room_growth_ratio=0.18,
    min_large_gap_shared_boundary_m=0.35,
):
    outline = get_plan_outline(plan_id)
    repaired = clip_prediction_to_outline(prediction_gdf, plan_id).copy()
    if outline.area <= 0:
        return repaired

    repaired = resolve_room_overlaps(repaired)
    max_added_area = outline.area * float(max_added_area_ratio)
    added_area = 0.0
    filler_rows = []

    # Iteratively recompute gaps after each assignment so newly expanded rooms can own nearby residual gaps.
    for _ in range(3):
        current_union = union_geometries(repaired.geometry) if len(repaired) else Polygon()
        gap_geom = outline.difference(current_union).buffer(0)
        gap_parts = sorted(geometry_polygon_parts(gap_geom), key=lambda geom: geom.area, reverse=True)
        changed = False

        for gap in gap_parts:
            if gap.area < filler_min_area_m2 or added_area + gap.area > max_added_area:
                continue
            owner_index = best_gap_owner_index(repaired, gap) if len(repaired) else None
            if owner_index is not None:
                owner_geom = repaired.at[owner_index, "geometry"]
                owner_area = max(float(owner_geom.area), 1e-6)
                shared_boundary = owner_geom.boundary.intersection(gap.boundary).length
                gap_is_small = gap.area <= small_gap_area_m2
                growth_is_reasonable = (gap.area / owner_area) <= max_room_growth_ratio
                boundary_is_strong = shared_boundary >= min_large_gap_shared_boundary_m
                if gap_is_small or (growth_is_reasonable and boundary_is_strong):
                    repaired.at[owner_index, "geometry"] = unary_union([owner_geom, gap]).buffer(0)
                    repaired.at[owner_index, "area_m2"] = repaired.at[owner_index, "geometry"].area
                    added_area += gap.area
                    changed = True
                    continue
            if allow_filler_rooms:
                filler_rows.append({
                    "room_index": len(repaired) + len(filler_rows),
                    "roomtype": filler_roomtype if filler_roomtype in ROOMTYPE_TO_ID else ROOMTYPE_ORDER[0],
                    "area_m2": gap.area,
                    "geometry": gap,
                })
                added_area += gap.area
                changed = True

        if not changed:
            break
        repaired = resolve_room_overlaps(repaired)

    if filler_rows:
        repaired = pd.concat([repaired, gpd.GeoDataFrame(filler_rows, geometry="geometry")], ignore_index=True)
    repaired = gpd.GeoDataFrame(repaired, geometry="geometry")
    repaired["geometry"] = repaired.geometry.map(lambda geom: geom.buffer(0))
    repaired = repaired.loc[~repaired.geometry.is_empty].copy()
    repaired["area_m2"] = repaired.geometry.area
    repaired = repaired.loc[repaired["area_m2"] > 0.05].copy()
    repaired = resolve_room_overlaps(repaired)

    final_union = union_geometries(repaired.geometry) if len(repaired) else Polygon()
    remaining_gap_area = outline.difference(final_union).area
    repaired.attrs.update(prediction_gdf.attrs)
    repaired.attrs["coverage_repaired"] = True
    repaired.attrs["coverage_added_area_m2"] = float(added_area)
    repaired.attrs["remaining_gap_area_m2"] = float(remaining_gap_area)
    return repaired


def refine_scaffold_with_denoiser(
    model,
    plan_id,
    scaffold_gdf,
    denoise_t=60,
    refinement_strength=0.25,
    preserve_scaffold_roomtypes=True,
    min_coverage_drop=-0.03,
    neural_refinement_enabled=False,
):
    if model is None or not neural_refinement_enabled:
        repaired = repair_outline_coverage(scaffold_gdf, plan_id)
        repaired.attrs.update(scaffold_gdf.attrs)
        repaired.attrs["refined_by"] = "coverage_repaired_scaffold"
        return repaired
    arrays = prediction_to_padded_arrays(scaffold_gdf, plan_id)
    batch = {
        "coords": torch.from_numpy(arrays["coords"])[None],
        "corner_mask": torch.from_numpy(arrays["corner_mask"])[None],
        "room_mask": torch.from_numpy(arrays["room_mask"])[None],
        "roomtype_ids": torch.from_numpy(arrays["roomtype_ids"])[None],
        "outline_cont": torch.from_numpy(arrays["outline_cont"])[None],
        "outline_mask": torch.from_numpy(arrays["outline_mask"])[None],
    }
    t = torch.tensor([denoise_t], dtype=torch.long)
    generator = torch.Generator().manual_seed(SEED + int(plan_id))

    model.eval()
    with torch.no_grad():
        noisy_coords, _ = add_diffusion_noise(batch["coords"], batch["corner_mask"], t, generator=generator)
        pred_noise, _, roomtype_logits = model(
            noisy_coords,
            batch["roomtype_ids"],
            batch["corner_mask"],
            batch["room_mask"],
            batch["outline_cont"],
            batch["outline_mask"],
            t,
        )
        alpha_bar = cosine_alpha_bar(t).view(1, 1, 1, 1).clamp(1e-5, 0.999)
        pred_x0 = (noisy_coords - (1.0 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt()
        scaffold_coords = batch["coords"]
        blended = scaffold_coords + float(refinement_strength) * (pred_x0 - scaffold_coords)
        blended = torch.where(batch["corner_mask"][..., None], blended, scaffold_coords)
        pred_grid = continuous_to_grid(blended.squeeze(0).numpy())

        if preserve_scaffold_roomtypes:
            pred_roomtype_ids = arrays["roomtype_ids"].copy()
        else:
            pred_roomtype_ids = roomtype_logits.squeeze(0).argmax(dim=-1).numpy()
            pred_roomtype_ids[~arrays["room_mask"]] = PAD_ROOMTYPE_ID

    refined = predicted_arrays_to_gdf(
        pred_grid,
        pred_roomtype_ids,
        arrays["corner_mask"],
        arrays["bounds"],
    )
    refined = clip_prediction_to_outline(refined, plan_id)
    repaired_refined = repair_outline_coverage(refined, plan_id)

    scaffold_coverage = outline_coverage(plan_id, scaffold_gdf)
    repaired_scaffold = repair_outline_coverage(scaffold_gdf, plan_id)
    repaired_scaffold_coverage = outline_coverage(plan_id, repaired_scaffold)
    repaired_refined_coverage = outline_coverage(plan_id, repaired_refined)
    room_count_ratio = len(repaired_refined) / max(len(scaffold_gdf), 1)

    if repaired_refined_coverage < scaffold_coverage + min_coverage_drop or room_count_ratio < 0.85:
        repaired_scaffold.attrs.update(scaffold_gdf.attrs)
        repaired_scaffold.attrs.update({
            "refined_by": "coverage_repaired_scaffold_after_guard",
            "guard_reason": "denoiser_coverage_or_room_count_drop",
            "scaffold_coverage": scaffold_coverage,
            "attempted_refined_coverage": repaired_refined_coverage,
            "repaired_scaffold_coverage": repaired_scaffold_coverage,
        })
        return repaired_scaffold

    repaired_refined.attrs.update({
        "source_plan_id": int(scaffold_gdf.attrs.get("source_plan_id", -1)),
        "sample_index": int(scaffold_gdf.attrs.get("sample_index", -1)),
        "refined_by": "denoiser_residual_plus_coverage_repair",
        "scaffold_coverage": scaffold_coverage,
        "attempted_refined_coverage": repaired_refined_coverage,
        "repaired_scaffold_coverage": repaired_scaffold_coverage,
    })
    return repaired_refined


def compare_scaffold_and_refined(plan_id=None, n_samples=2, neural_refinement_enabled=False):
    if plan_id is None:
        plan_id = int(val_plan_ids[0])
    model = load_outline_denoiser() if neural_refinement_enabled else None
    if neural_refinement_enabled and model is None:
        model, _ = train_outline_denoiser()

    scaffolds, _, _, _ = predict_rooms_program_aware_set(plan_id, n_samples=n_samples)
    rows = []
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 4.5 * n_samples))
    axes = np.asarray(axes).reshape(n_samples, 3)
    outline = get_plan_outline(plan_id)
    target = get_plan_rooms(plan_id)

    for row_index, scaffold in enumerate(scaffolds):
        refined = refine_scaffold_with_denoiser(model, plan_id, scaffold)
        plot_room_polygons(axes[row_index, 0], target, outline, f"Ground truth\nplan_id={plan_id}")
        plot_room_polygons(axes[row_index, 1], scaffold, outline, f"Program scaffold\nrooms={len(scaffold)}")
        plot_room_polygons(axes[row_index, 2], refined, outline, f"Coverage-repaired final\nrooms={len(refined)}")

        scaffold_quality = prediction_quality(plan_id, scaffold).add_prefix("scaffold_")
        refined_quality = prediction_quality(plan_id, refined).add_prefix("refined_")
        diagnostics = pd.Series({
            "refined_by": refined.attrs.get("refined_by", ""),
            "coverage_added_area_m2": float(refined.attrs.get("coverage_added_area_m2", 0.0)),
            "remaining_gap_area_m2": float(refined.attrs.get("remaining_gap_area_m2", np.nan)),
            "overlap_removed_area_m2": float(refined.attrs.get("overlap_removed_area_m2", 0.0)),
            "scaffold_naturalness_penalty": room_naturalness_penalty(scaffold),
            "refined_naturalness_penalty": room_naturalness_penalty(refined),
        })
        rows.append(pd.concat([scaffold_quality, refined_quality, diagnostics]))

    plt.tight_layout()
    plt.savefig(f"scaffold_vs_refined_{plan_id}.png")
    plt.close(fig)
    comparison_df = pd.DataFrame(rows)
    display(comparison_df[[
        "scaffold_predicted_rooms",
        "refined_predicted_rooms",
        "scaffold_union_iou",
        "refined_union_iou",
        "scaffold_prediction_outline_coverage",
        "refined_prediction_outline_coverage",
        "scaffold_mean_best_same_type_iou",
        "refined_mean_best_same_type_iou",
        "coverage_added_area_m2",
        "remaining_gap_area_m2",
        "overlap_removed_area_m2",
        "scaffold_naturalness_penalty",
        "refined_naturalness_penalty",
    ]].round(3))
    return comparison_df


if RUN_DENOISER_TRAINING:
    trained_model, training_history = train_outline_denoiser()
else:
    trained_model = load_outline_denoiser()
    training_history = pd.DataFrame()

# Manual save/load examples:
# save_outline_denoiser_weights(trained_model, CHECKPOINT_PATH, history=training_history.to_dict("records"))
# trained_model = load_outline_denoiser(CHECKPOINT_PATH)
# checkpoint = load_outline_denoiser_weights(trained_model, CHECKPOINT_PATH)

comparison_df = compare_scaffold_and_refined(n_samples=4, neural_refinement_enabled=ENABLE_NEURAL_REFINEMENT)

if trained_model is not None:
    save_outline_denoiser_weights(trained_model, CHECKPOINT_PATH, history=training_history.to_dict("records"))
    trained_model = load_outline_denoiser(CHECKPOINT_PATH)
    checkpoint = load_outline_denoiser_weights(trained_model, CHECKPOINT_PATH)

