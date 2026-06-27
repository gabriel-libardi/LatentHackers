import pandas as pd
import geopandas as gpd
from shapely import wkt
import torch
from torch.utils.data import Dataset
import numpy as np

ROOM_TYPES = ['Bathroom', 'Livingroom', 'Bedroom', 'Kitchen', 'Balcony', 'Corridor', 'Dining', 'Structure', 'Stairs', 'Storeroom']
ROOM_TYPE_TO_IDX = {name: idx for idx, name in enumerate(ROOM_TYPES)}

class FloorplanDataset(Dataset):
    def __init__(self, csv_path, index_file, max_total_corners=800, max_outline_len=128):
        self.max_total_corners = max_total_corners
        self.max_outline_len = max_outline_len
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        with open(index_file, "r") as f:
            self.floor_ids = [int(line.strip()) for line in f if line.strip()]
        print(f"Loaded {len(self.floor_ids)} floor IDs for this split.")
        floor_ids_set = set(self.floor_ids)
        self.df = df[df['floor_id'].isin(floor_ids_set)].copy()
        self.grouped = self.df.groupby('floor_id')
        self.floor_ids = [fid for fid in self.floor_ids if fid in self.grouped.groups]
        print(f"Active floor plans after intersection: {len(self.floor_ids)}")

    def __len__(self):
        return len(self.floor_ids)

    def __getitem__(self, idx):
        floor_id = self.floor_ids[idx]
        floor_df = self.grouped.get_group(floor_id)
        area_df = floor_df[floor_df['entity_type'] == 'area'].copy()
        if len(area_df) == 0:
            return self.__getitem__((idx + 1) % len(self))
        geoms = []
        room_types = []
        for _, row in area_df.iterrows():
            try:
                geom = wkt.loads(row['geom'])
                if geom.geom_type in ['Polygon', 'MultiPolygon']:
                    geoms.append(geom)
                    room_types.append(row['roomtype'])
            except Exception:
                continue
        if len(geoms) == 0:
            return self.__getitem__((idx + 1) % len(self))
        buffered = [g.buffer(0.3) for g in geoms]
        try:
            union_geom = buffered[0]
            for g in buffered[1:]:
                union_geom = union_geom.union(g)
            outline = union_geom.buffer(-0.3)
        except Exception:
            return self.__getitem__((idx + 1) % len(self))
        if outline.is_empty:
            return self.__getitem__((idx + 1) % len(self))
        minx, miny, maxx, maxy = outline.bounds
        max_size = max(maxx - minx, maxy - miny)
        if max_size == 0:
            max_size = 1.0
        def norm(x, y):
            return (x - minx) / max_size, (y - miny) / max_size
        outline_pts = []
        if outline.geom_type == 'Polygon':
            outline_pts = list(outline.exterior.coords)[:-1]
        elif outline.geom_type == 'MultiPolygon':
            for p in outline.geoms:
                outline_pts.extend(list(p.exterior.coords)[:-1])
        outline_norm = [norm(p[0], p[1]) for p in outline_pts]
        if len(outline_norm) > self.max_outline_len:
            indices = np.linspace(0, len(outline_norm) - 1, self.max_outline_len, dtype=int)
            outline_norm = [outline_norm[i] for i in indices]
        corners = []
        entity_types = []
        entity_idxs = []
        corner_idxs = []
        for r_idx, (geom, r_type) in enumerate(zip(geoms, room_types)):
            t_idx = ROOM_TYPE_TO_IDX.get(r_type, 0)
            pts = []
            if geom.geom_type == 'Polygon':
                pts = list(geom.exterior.coords)[:-1]
            elif geom.geom_type == 'MultiPolygon':
                for p in geom.geoms:
                    pts.extend(list(p.exterior.coords)[:-1])
            for c_idx, p in enumerate(pts):
                cx, cy = norm(p[0], p[1])
                corners.append((cx, cy))
                entity_types.append(t_idx)
                entity_idxs.append(r_idx)
                corner_idxs.append(c_idx)
        if len(corners) > self.max_total_corners:
            return self.__getitem__((idx + 1) % len(self))
        return {"x_0": torch.tensor(corners, dtype=torch.float32), "entity_type": torch.tensor(entity_types, dtype=torch.long), "entity_idx": torch.tensor(entity_idxs, dtype=torch.long), "corner_idx": torch.tensor(corner_idxs, dtype=torch.long), "outline": torch.tensor(outline_norm, dtype=torch.float32), "floor_id": floor_id, "minx": minx, "miny": miny, "max_size": max_size}

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return {}
    B = len(batch)
    lengths = [item["x_0"].shape[0] for item in batch]
    max_len = max(lengths)
    x_0_padded = torch.zeros(B, max_len, 2)
    entity_type_padded = torch.zeros(B, max_len, dtype=torch.long)
    entity_idx_padded = torch.zeros(B, max_len, dtype=torch.long)
    corner_idx_padded = torch.zeros(B, max_len, dtype=torch.long)
    corner_mask = torch.zeros(B, max_len, dtype=torch.float32)
    for i, item in enumerate(batch):
        l = lengths[i]
        x_0_padded[i, :l] = item["x_0"]
        entity_type_padded[i, :l] = item["entity_type"]
        entity_idx_padded[i, :l] = item["entity_idx"]
        corner_idx_padded[i, :l] = item["corner_idx"]
        corner_mask[i, :l] = 1.0
    out_lengths = [item["outline"].shape[0] for item in batch]
    max_out_len = max(out_lengths)
    outline_padded = torch.zeros(B, max_out_len, 2)
    outline_mask = torch.zeros(B, max_out_len, dtype=torch.float32)
    for i, item in enumerate(batch):
        ol = out_lengths[i]
        outline_padded[i, :ol] = item["outline"]
        outline_mask[i, :ol] = 1.0
    return {"x_0": x_0_padded, "entity_type": entity_type_padded, "entity_idx": entity_idx_padded, "corner_idx": corner_idx_padded, "corner_mask": corner_mask, "outline": outline_padded, "outline_mask": outline_mask, "floor_ids": [item["floor_id"] for item in batch], "minxs": [item["minx"] for item in batch], "minys": [item["miny"] for item in batch], "max_sizes": [item["max_size"] for item in batch]}
