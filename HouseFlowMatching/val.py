import argparse
import os
import random
import sys
import numpy as np
import torch
import geopandas as gpd
from shapely import wkt
import shapely.affinity
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from dataset import FloorplanDataset, ROOM_TYPES
from house_flow import HouseFlowModel, FlowMatching

def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

class OutlineConditionedGenerator:
    def __init__(self, csv_path, train_index, model_path, d_model=256, num_layers=6, sample_steps=100, device="cpu"):
        self.device = torch.device(device)
        self.max_outline_len = 128
        print("Building query database from training set...")
        self.train_dataset = FloorplanDataset(csv_path, train_index)
        self.train_outlines = []
        for i in range(len(self.train_dataset)):
            try:
                item = self.train_dataset[i]
                if item is not None:
                    floor_id = item["floor_id"]
                    floor_df = self.train_dataset.grouped.get_group(floor_id)
                    area_df = floor_df[floor_df['entity_type'] == 'area'].copy()
                    geoms = [wkt.loads(g) for g in area_df['geom'] if wkt.loads(g).geom_type in ['Polygon', 'MultiPolygon']]
                    buffered = [g.buffer(0.3) for g in geoms]
                    union_geom = buffered[0]
                    for g in buffered[1:]:
                        union_geom = union_geom.union(g)
                    outline = union_geom.buffer(-0.3)
                    if not outline.is_empty:
                        self.train_outlines.append({"floor_id": floor_id, "geom": outline, "area": outline.area, "entity_type": item["entity_type"], "entity_idx": item["entity_idx"], "corner_idx": item["corner_idx"]})
            except Exception:
                continue
        print(f"Database built with {len(self.train_outlines)} valid training outlines.")
        self.model = HouseFlowModel(d_model=d_model, num_layers=num_layers, num_room_types=10, max_corners_per_room=512, max_rooms=512, max_outline_len=self.max_outline_len).to(self.device)
        if os.path.exists(model_path):
            print(f"Loading model weights from {model_path}...")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print(f"Warning: Model weights path '{model_path}' not found! Running with randomized weights.")
        self.model.eval()
        self.flow = FlowMatching(steps=sample_steps, device=self.device)
    def find_nearest_layout(self, query):
        best_iou = -1.0
        best = None
        q_area = query.area
        q_c = query.centroid
        cands = [o for o in self.train_outlines if 0.5 * q_area <= o["area"] <= 2.0 * q_area]
        if not cands:
            cands = self.train_outlines
        for c in cands:
            g = c["geom"]
            dx = q_c.x - g.centroid.x
            dy = q_c.y - g.centroid.y
            ga = shapely.affinity.translate(g, xoff=dx, yoff=dy)
            try:
                inter = query.intersection(ga).area
                uni = query.union(ga).area
                iou = inter / (uni + 1e-8)
            except Exception:
                iou = 0.0
            if iou > best_iou:
                best_iou = iou
                best = c
        return best
    def generate(self, outline_polygon, seed=42):
        set_seed(seed)
        nearest = self.find_nearest_layout(outline_polygon)
        if nearest is None:
            raise ValueError("No matching layout configuration found in the training database.")
        entity_type = nearest["entity_type"].to(self.device).unsqueeze(0)
        entity_idx = nearest["entity_idx"].to(self.device).unsqueeze(0)
        corner_idx = nearest["corner_idx"].to(self.device).unsqueeze(0)
        N = entity_type.shape[1]
        minx, miny, maxx, maxy = outline_polygon.bounds
        max_size = max(maxx - minx, maxy - miny)
        if max_size == 0:
            max_size = 1.0
        def norm(x, y):
            return (x - minx) / max_size, (y - miny) / max_size
        pts = []
        if outline_polygon.geom_type == 'Polygon':
            pts = list(outline_polygon.exterior.coords)[:-1]
        elif outline_polygon.geom_type == 'MultiPolygon':
            for p in outline_polygon.geoms:
                pts.extend(list(p.exterior.coords)[:-1])
        outline_norm = [norm(p[0], p[1]) for p in pts]
        if len(outline_norm) > self.max_outline_len:
            idx = np.linspace(0, len(outline_norm) - 1, self.max_outline_len, dtype=int)
            outline_norm = [outline_norm[i] for i in idx]
        outline_tensor = torch.tensor(outline_norm, dtype=torch.float32, device=self.device).unsqueeze(0)
        outline_mask = torch.ones(1, len(outline_norm), dtype=torch.float32, device=self.device)
        cond = {"entity_type": entity_type, "entity_idx": entity_idx, "corner_idx": corner_idx, "outline": outline_tensor, "outline_mask": outline_mask}
        sampled = self.flow.sample_loop(self.model, (1, N, 2), cond).squeeze(0).cpu().numpy()
        corners = [(cx * max_size + minx, cy * max_size + miny) for cx, cy in sampled]
        rooms = {}
        ei = entity_idx.squeeze(0).cpu().numpy()
        et = entity_type.squeeze(0).cpu().numpy()
        ci = corner_idx.squeeze(0).cpu().numpy()
        for i in range(N):
            r = int(ei[i])
            if r not in rooms:
                rooms[r] = {"coords": [], "ci": [], "type": int(et[i])}
            rooms[r]["coords"].append(corners[i])
            rooms[r]["ci"].append(int(ci[i]))
        out = []
        for r, room in rooms.items():
            sorted_pts = [c for _, c in sorted(zip(room["ci"], room["coords"]))]
            if len(sorted_pts) >= 3:
                try:
                    poly = Polygon(sorted_pts)
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                    out.append((poly, ROOM_TYPES[room["type"]]))
                except Exception:
                    continue
        return out

def validate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = OutlineConditionedGenerator(csv_path=args.csv_path, train_index=args.train_index, model_path=args.model_path, sample_steps=args.sample_steps, device=device)
    test_dataset = FloorplanDataset(args.csv_path, args.test_index)
    output_dir = "val_results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved in: {output_dir}")
    num = min(5, len(test_dataset))
    for i in range(num):
        item = test_dataset[i]
        if item is None:
            continue
        floor_id = item["floor_id"]
        print(f"Processing plan validation for floor ID {floor_id}...")
        floor_df = test_dataset.grouped.get_group(floor_id)
        area_df = floor_df[floor_df['entity_type'] == 'area'].copy()
        gt_rooms = []
        geoms = []
        for _, row in area_df.iterrows():
            try:
                g = wkt.loads(row['geom'])
                if g.geom_type in ['Polygon', 'MultiPolygon']:
                    geoms.append(g)
                    gt_rooms.append((g, row['roomtype']))
            except Exception:
                continue
        buffered = [g.buffer(0.3) for g in geoms]
        union_geom = buffered[0]
        for g in buffered[1:]:
            union_geom = union_geom.union(g)
        outline_geom = union_geom.buffer(-0.3)
        try:
            pred_rooms = generator.generate(outline_geom)
        except Exception as e:
            print(f"Error generating layout for floor {floor_id}: {e}")
            continue
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
        gpd.GeoDataFrame(geometry=[outline_geom]).plot(ax=ax1, facecolor='#f4f4f4', edgecolor='black', linewidth=3)
        ax1.set_title("Input: Apartment Outer Outline", fontsize=14, fontweight='bold')
        ax1.axis('equal'); ax1.axis('off')
        pred_gdf = gpd.GeoDataFrame(geometry=[r[0] for r in pred_rooms], data={"roomtype": [r[1] for r in pred_rooms]})
        pred_gdf.plot(ax=ax2, column="roomtype", legend=True, cmap='Set3', edgecolor='white', linewidth=1.5)
        gpd.GeoDataFrame(geometry=[outline_geom]).plot(ax=ax2, facecolor='none', edgecolor='black', linewidth=2, alpha=0.4)
        ax2.set_title("Predicted Layout (HouseFlowMatching)", fontsize=14, fontweight='bold')
        ax2.axis('equal'); ax2.axis('off')
        gt_gdf = gpd.GeoDataFrame(geometry=[r[0] for r in gt_rooms], data={"roomtype": [r[1] for r in gt_rooms]})
        gt_gdf.plot(ax=ax3, column="roomtype", legend=True, cmap='Set3', edgecolor='white', linewidth=1.5)
        gpd.GeoDataFrame(geometry=[outline_geom]).plot(ax=ax3, facecolor='none', edgecolor='black', linewidth=2, alpha=0.4)
        ax3.set_title("Ground Truth Layout", fontsize=14, fontweight='bold')
        ax3.axis('equal'); ax3.axis('off')
        plt.suptitle(f"Layout Generation Comparison (Floor ID: {floor_id})", fontsize=16)
        plt.tight_layout()
        path = os.path.join(output_dir, f"comparison_{floor_id}.png")
        plt.savefig(path)
        plt.close()
        print(f"Comparison plot saved successfully to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Outline-Conditioned HouseFlowMatching Model")
    parser.add_argument("--csv_path", type=str, default="mds_V2_5.372k.csv")
    parser.add_argument("--train_index", type=str, default="train_indices.txt")
    parser.add_argument("--test_index", type=str, default="test_indices.txt")
    parser.add_argument("--model_path", type=str, default="flow_final.pth")
    parser.add_argument("--sample_steps", type=int, default=100)
    if len(sys.argv) == 1:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()
    validate(args)
