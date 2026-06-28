import os
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Polygon, MultiPolygon
import torch
from torch.utils.data import Dataset
import numpy as np

# Room mappings and names from constants.py
ROOM_MAPPING = {
    'ROOM': 'Bedroom',
    'BEDROOM': 'Bedroom',
    'BALCONY': 'Balcony',
    'TERRACE': 'Balcony',
    'KITCHEN': 'Kitchen',
    'DINING': 'Dining',
    'KITCHEN_DINING': 'Kitchen',
    'LIVING_ROOM': 'Livingroom',
    'LIVING_DINING': 'Livingroom',
    'BATHROOM': 'Bathroom',
    'CORRIDOR': 'Corridor',
    'CORRIDORS_AND_HALLS': 'Corridor',
    'STAIRS': 'Stairs',
    'STAIRCASE': 'Stairs',
    'ELEVATOR': 'Stairs',
    'RAILING': 'Balcony',
    'VOID': 'Stairs',
    'SHAFT': 'Structure',
    'WALL': 'Structure',
    'COLUMN': 'Structure',
    'STOREROOM': 'Storeroom',
    'ENTRANCE_DOOR': 'Entrance Door',
    'DOOR': 'Door',
    'WINDOW': 'Window'
}

ROOM_NAMES = [
    'Bedroom', 'Livingroom', 'Kitchen', 'Dining', 'Corridor', 
    'Stairs', 'Storeroom', 'Bathroom', 'Balcony', 'Structure', 
    'Door', 'Entrance Door', 'Window'
]
ROOM_TYPES = ROOM_NAMES
ROOM_TYPE_TO_IDX = {name: idx for idx, name in enumerate(ROOM_NAMES)}

def generate_outline(room_geometries, wall_bridge_distance=0.3):
    """
    Outline generation function integrated from dataExploration notebook.
    """
    return room_geometries.buffer(wall_bridge_distance).union_all().buffer(-wall_bridge_distance)

class FloorplanDataset(Dataset):
    def __init__(self, csv_path, index_file, max_total_corners=800, max_outline_len=128):
        self.max_total_corners = max_total_corners
        self.max_outline_len = max_outline_len
        
        # Load the CSV
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Load indices defining the split
        with open(index_file, "r") as f:
            self.floor_ids = [int(line.strip()) for line in f if line.strip()]
            
        print(f"Loaded {len(self.floor_ids)} floor IDs for this split.")
        
        # Filter the dataframe to keep only the floor plans in this split
        floor_ids_set = set(self.floor_ids)
        self.df = df[df['floor_id'].isin(floor_ids_set)].copy()
        
        # Pre-group by floor_id to make loading fast
        self.grouped = self.df.groupby('floor_id')
        self.floor_ids = [fid for fid in self.floor_ids if fid in self.grouped.groups]
        print(f"Active floor plans after intersection: {len(self.floor_ids)}")
        
    def __len__(self):
        return len(self.floor_ids)
        
    def __getitem__(self, idx):
        floor_id = self.floor_ids[idx]
        floor_df = self.grouped.get_group(floor_id)
        
        # Filter area rows
        area_df = floor_df[floor_df['entity_type'] == 'area'].copy()
        if len(area_df) == 0:
            # Fallback in case of empty plan
            return self.__getitem__((idx + 1) % len(self))
            
        # Parse geometries
        geoms = []
        room_types = []
        for _, row in area_df.iterrows():
            try:
                geom = wkt.loads(row['geom'])
                # Only keep Polygons and MultiPolygons
                if geom.geom_type in ['Polygon', 'MultiPolygon']:
                    geoms.append(geom)
                    
                    # Apply mapping from dataExploration
                    subtype_str = str(row['entity_subtype']).strip()
                    room_name = ROOM_MAPPING.get(subtype_str, 'Livingroom')
                    room_types.append(room_name)
            except Exception:
                continue
                
        if len(geoms) == 0:
            return self.__getitem__((idx + 1) % len(self))
            
        # Build the outer outline using the notebook's contour logic
        try:
            room_series = gpd.GeoSeries(geoms)
            solid_outline_geom = generate_outline(room_series, wall_bridge_distance=0.3)
        except Exception:
            return self.__getitem__((idx + 1) % len(self))
            
        if solid_outline_geom.is_empty:
            return self.__getitem__((idx + 1) % len(self))
            
        # Normalize coordinates to [0, 1] relative to the outline's bounding box
        minx, miny, maxx, maxy = solid_outline_geom.bounds
        width = maxx - minx
        height = maxy - miny
        max_size = max(width, height)
        if max_size == 0:
            max_size = 1.0
            
        def normalize_pt(x, y):
            return (x - minx) / max_size, (y - miny) / max_size
            
        # Extract outline coordinates
        outline_pts = []
        if solid_outline_geom.geom_type == 'Polygon':
            outline_pts = list(solid_outline_geom.exterior.coords)[:-1]
        elif solid_outline_geom.geom_type == 'MultiPolygon':
            # Collect from all parts
            for p in solid_outline_geom.geoms:
                outline_pts.extend(list(p.exterior.coords)[:-1])
                
        # Normalize outline
        outline_norm = [normalize_pt(pt[0], pt[1]) for pt in outline_pts]
        if len(outline_norm) > self.max_outline_len:
            # Downsample to max_outline_len
            indices = np.linspace(0, len(outline_norm) - 1, self.max_outline_len, dtype=int)
            outline_norm = [outline_norm[i] for i in indices]
            
        # Extract room corners
        corners = []
        entity_types = []
        entity_idxs = []
        corner_idxs = []
        
        for r_idx, (geom, r_type) in enumerate(zip(geoms, room_types)):
            t_idx = ROOM_TYPE_TO_IDX.get(r_type, 0)
            
            # Extract vertices
            pts = []
            if geom.geom_type == 'Polygon':
                pts = list(geom.exterior.coords)[:-1]
            elif geom.geom_type == 'MultiPolygon':
                for p in geom.geoms:
                    pts.extend(list(p.exterior.coords)[:-1])
                    
            for c_idx, pt in enumerate(pts):
                cx, cy = normalize_pt(pt[0], pt[1])
                corners.append((cx, cy))
                entity_types.append(t_idx)
                entity_idxs.append(r_idx)
                corner_idxs.append(c_idx)
                
        # Check if total corners exceeds limit
        if len(corners) > self.max_total_corners:
            # Skip this plan to avoid excessive memory usage
            return self.__getitem__((idx + 1) % len(self))
            
        return {
            "x_0": torch.tensor(corners, dtype=torch.float32),
            "entity_type": torch.tensor(entity_types, dtype=torch.long),
            "entity_idx": torch.tensor(entity_idxs, dtype=torch.long),
            "corner_idx": torch.tensor(corner_idxs, dtype=torch.long),
            "outline": torch.tensor(outline_norm, dtype=torch.float32),
            "floor_id": floor_id,
            "minx": minx,
            "miny": miny,
            "max_size": max_size
        }

def collate_fn(batch):
    # Filter out None values in case of loading issues
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return {}
        
    B = len(batch)
    
    # Pad room corners
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
        
    # Pad outlines
    out_lengths = [item["outline"].shape[0] for item in batch]
    max_out_len = max(out_lengths)
    
    outline_padded = torch.zeros(B, max_out_len, 2)
    outline_mask = torch.zeros(B, max_out_len, dtype=torch.float32)
    
    for i, item in enumerate(batch):
        ol = out_lengths[i]
        outline_padded[i, :ol] = item["outline"]
        outline_mask[i, :ol] = 1.0
        
    floor_ids = [item["floor_id"] for item in batch]
    minxs = [item["minx"] for item in batch]
    minys = [item["miny"] for item in batch]
    max_sizes = [item["max_size"] for item in batch]
    
    return {
        "x_0": x_0_padded,
        "entity_type": entity_type_padded,
        "entity_idx": entity_idx_padded,
        "corner_idx": corner_idx_padded,
        "corner_mask": corner_mask,
        "outline": outline_padded,
        "outline_mask": outline_mask,
        "floor_ids": floor_ids,
        "minxs": minxs,
        "minys": minys,
        "max_sizes": max_sizes
    }
