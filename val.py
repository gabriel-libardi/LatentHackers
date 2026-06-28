import argparse
import os
import torch
import pandas as pd
import geopandas as gpd
from shapely import wkt
import shapely.affinity
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
import numpy as np

from dataset import FloorplanDataset, ROOM_TYPES, ROOM_TYPE_TO_IDX, ROOM_MAPPING
from house_diffusion import HouseDiffusionModel, GaussianDiffusion

class OutlineConditionedGenerator:
    """
    Retrieves the closest training layout's room configuration and uses the trained
    HouseDiffusion model to generate the room partitions for a given bare outline.
    """
    def __init__(self, csv_path, train_index, model_path, d_model=256, num_layers=6, device="cpu"):
        self.device = torch.device(device)
        self.max_outline_len = 128
        
        # Load train set to use as database for query retrieval
        print("Building query database from training set...")
        self.train_dataset = FloorplanDataset(csv_path, train_index)
        
        # Build training outlines database
        self.train_outlines = []
        for i in range(len(self.train_dataset)):
            try:
                item = self.train_dataset[i]
                if item is not None:
                    # Construct shapely polygon from raw geometry
                    floor_id = item["floor_id"]
                    floor_df = self.train_dataset.grouped.get_group(floor_id)
                    area_df = floor_df[floor_df['entity_type'] == 'area'].copy()
                    
                    geoms = [wkt.loads(g) for g in area_df['geom'] if wkt.loads(g).geom_type in ['Polygon', 'MultiPolygon']]
                    wall_bridge_distance = 0.3
                    buffered_geoms = [g.buffer(wall_bridge_distance) for g in geoms]
                    union_geom = buffered_geoms[0]
                    for g in buffered_geoms[1:]:
                        union_geom = union_geom.union(g)
                    solid_outline_geom = union_geom.buffer(-wall_bridge_distance)
                    
                    if not solid_outline_geom.is_empty:
                        self.train_outlines.append({
                            "floor_id": floor_id,
                            "geom": solid_outline_geom,
                            "area": solid_outline_geom.area,
                            "entity_type": item["entity_type"],
                            "entity_idx": item["entity_idx"],
                            "corner_idx": item["corner_idx"]
                        })
            except Exception:
                continue
        print(f"Database built with {len(self.train_outlines)} valid training outlines.")
        
        # Initialize model
        self.model = HouseDiffusionModel(
            d_model=d_model,
            num_layers=num_layers,
            num_room_types=10,
            max_corners_per_room=512,
            max_rooms=512,
            max_outline_len=self.max_outline_len
        ).to(self.device)
        
        if os.path.exists(model_path):
            print(f"Loading model weights from {model_path}...")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            print(f"Warning: Model weights path '{model_path}' not found! Running with randomized weights.")
            
        self.model.eval()
        self.diffusion = GaussianDiffusion(steps=1000, device=self.device)
        
    def find_nearest_layout(self, query_outline_geom, probabilistic=True, top_k=10):
        """
        Finds the training outline most similar to the query outline in area and shape
        using centroid-aligned Intersection over Union (IoU).
        If probabilistic is True, randomly samples from the top_k best matches to serve as
        a probabilistic prior for the room configuration.
        """
        query_area = query_outline_geom.area
        query_centroid = query_outline_geom.centroid
        
        # Filter database by area to make query matching fast
        candidate_matches = [
            out for out in self.train_outlines
            if 0.5 * query_area <= out["area"] <= 2.0 * query_area
        ]
        
        if not candidate_matches:
            candidate_matches = self.train_outlines
            
        matches = []
        for candidate in candidate_matches:
            train_geom = candidate["geom"]
            
            # Align centroids
            dx = query_centroid.x - train_geom.centroid.x
            dy = query_centroid.y - train_geom.centroid.y
            aligned_train_geom = shapely.affinity.translate(train_geom, xoff=dx, yoff=dy)
            
            # Compute centroid-aligned IoU
            try:
                intersection = query_outline_geom.intersection(aligned_train_geom).area
                union = query_outline_geom.union(aligned_train_geom).area
                iou = intersection / (union + 1e-8)
            except Exception:
                iou = 0.0
                
            matches.append((iou, candidate))
            
        # Sort by IoU in descending order
        matches.sort(key=lambda x: x[0], reverse=True)
        
        if not matches:
            return None
            
        if probabilistic:
            # Sample from the top K matches
            k = min(top_k, len(matches))
            top_candidates = matches[:k]
            # Randomly select one
            selected_idx = np.random.randint(0, k)
            return top_candidates[selected_idx][1]
        else:
            return matches[0][1]

    def generate(self, outline_polygon, probabilistic=True, top_k=10):
        """
        Main entry point for generating room partitions given a bare apartment outline.
        Returns:
            List of tuples: (shapely.geometry.Polygon, room_type_name)
        """
        # 1. Retrieve the best matching room configuration layout
        nearest = self.find_nearest_layout(outline_polygon, probabilistic=probabilistic, top_k=top_k)
        if nearest is None:
            raise ValueError("No matching layout configuration found in the training database.")
            
        # Get query indices from nearest training example
        entity_type = nearest["entity_type"].to(self.device).unsqueeze(0)  # (1, N)
        entity_idx = nearest["entity_idx"].to(self.device).unsqueeze(0)    # (1, N)
        corner_idx = nearest["corner_idx"].to(self.device).unsqueeze(0)    # (1, N)
        N = entity_type.shape[1]
        
        # 2. Normalize outline polygon coordinates
        minx, miny, maxx, maxy = outline_polygon.bounds
        width = maxx - minx
        height = maxy - miny
        max_size = max(width, height)
        if max_size == 0:
            max_size = 1.0
            
        def normalize_pt(x, y):
            return (x - minx) / max_size, (y - miny) / max_size
            
        outline_pts = []
        if outline_polygon.geom_type == 'Polygon':
            outline_pts = list(outline_polygon.exterior.coords)[:-1]
        elif outline_polygon.geom_type == 'MultiPolygon':
            for p in outline_polygon.geoms:
                outline_pts.extend(list(p.exterior.coords)[:-1])
                
        outline_norm = [normalize_pt(pt[0], pt[1]) for pt in outline_pts]
        if len(outline_norm) > self.max_outline_len:
            indices = np.linspace(0, len(outline_norm) - 1, self.max_outline_len, dtype=int)
            outline_norm = [outline_norm[i] for i in indices]
            
        outline_tensor = torch.tensor(outline_norm, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, M, 2)
        outline_mask = torch.ones(1, len(outline_norm), dtype=torch.float32, device=self.device)          # (1, M)
        
        cond = {
            "entity_type": entity_type,
            "entity_idx": entity_idx,
            "corner_idx": corner_idx,
            "outline": outline_tensor,
            "outline_mask": outline_mask,
            "corner_mask": torch.ones_like(entity_type)
        }
        
        # 3. Denoise with diffusion model (sampling loop)
        print("Denoising coordinates with HouseDiffusion...")
        x_shape = (1, N, 2)
        sampled_normalized = self.diffusion.p_sample_loop(self.model, x_shape, cond)  # (1, N, 2)
        sampled_normalized = sampled_normalized.squeeze(0).cpu().numpy()              # (N, 2)
        
        # 4. De-normalize generated coordinates back to original scale
        generated_corners = []
        for cx, cy in sampled_normalized:
            gx = cx * max_size + minx
            gy = cy * max_size + miny
            generated_corners.append((gx, gy))
            
        # 5. Reconstruct room polygons
        rooms = {}
        entity_idx_np = entity_idx.squeeze(0).cpu().numpy()
        entity_type_np = entity_type.squeeze(0).cpu().numpy()
        corner_idx_np = corner_idx.squeeze(0).cpu().numpy()
        
        for i in range(N):
            ent_id = entity_idx_np[i]
            if ent_id not in rooms:
                rooms[ent_id] = {
                    "coords": [],
                    "corner_indices": [],
                    "type_idx": entity_type_np[i]
                }
            rooms[ent_id]["coords"].append(generated_corners[i])
            rooms[ent_id]["corner_indices"].append(corner_idx_np[i])
            
        generated_polygons = []
        for ent_id, room in rooms.items():
            # Sort the coordinates to match the correct corner sequence
            sorted_pts = [c for _, c in sorted(zip(room["corner_indices"], room["coords"]))]
            if len(sorted_pts) >= 3:
                try:
                    poly = Polygon(sorted_pts)
                    # Check validity; if invalid, try buffer(0)
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                    room_name = ROOM_TYPES[room["type_idx"]]
                    generated_polygons.append((poly, room_name))
                except Exception:
                    continue
                    
        return generated_polygons


def validate(args):
    # Select device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize Generator
    generator = OutlineConditionedGenerator(
        csv_path=args.csv_path,
        train_index=args.train_index,
        model_path=args.model_path,
        device=device
    )
    
    # Load test/val split dataset
    test_dataset = FloorplanDataset(args.csv_path, args.test_index)
    
    # Run generation on a few validation floorplans and save visualizations
    output_dir = "val_results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved in: {output_dir}")
    
    # Validate on first 5 samples
    num_to_validate = min(5, len(test_dataset))
    for i in range(num_to_validate):
        item = test_dataset[i]
        if item is None:
            continue
            
        floor_id = item["floor_id"]
        print(f"\nProcessing plan validation for floor ID {floor_id}...")
        
        # Load ground truth geometries
        floor_df = test_dataset.grouped.get_group(floor_id)
        area_df = floor_df[floor_df['entity_type'] == 'area'].copy()
        
        gt_rooms = []
        geoms = []
        for _, row in area_df.iterrows():
            try:
                g = wkt.loads(row['geom'])
                if g.geom_type in ['Polygon', 'MultiPolygon']:
                    geoms.append(g)
                    subtype_str = str(row['entity_subtype']).strip()
                    room_name = ROOM_MAPPING.get(subtype_str, 'Livingroom')
                    gt_rooms.append((g, room_name))
            except Exception:
                continue
                
        # Reconstruct ground truth outline using standard buffering logic
        wall_bridge_distance = 0.3
        buffered_geoms = [g.buffer(wall_bridge_distance) for g in geoms]
        union_geom = buffered_geoms[0]
        for g in buffered_geoms[1:]:
            union_geom = union_geom.union(g)
        outline_geom = union_geom.buffer(-wall_bridge_distance)
        
        # Generate room layout from bare outline using trained model
        try:
            pred_rooms = generator.generate(outline_geom)
        except Exception as e:
            print(f"Error generating layout for floor {floor_id}: {e}")
            continue
            
        # Plot and save comparison
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
        
        # 1. Input Outline
        gpd.GeoDataFrame(geometry=[outline_geom]).plot(ax=ax1, facecolor='#f4f4f4', edgecolor='black', linewidth=3)
        ax1.set_title("Input: Apartment Outer Outline", fontsize=14, fontweight='bold')
        ax1.axis('equal')
        ax1.axis('off')
        
        # 2. Predicted Rooms
        pred_gdf = gpd.GeoDataFrame(
            geometry=[r[0] for r in pred_rooms],
            data={"roomtype": [r[1] for r in pred_rooms]}
        )
        pred_gdf.plot(ax=ax2, column="roomtype", legend=True, cmap='Set3', edgecolor='white', linewidth=1.5)
        gpd.GeoDataFrame(geometry=[outline_geom]).plot(ax=ax2, facecolor='none', edgecolor='black', linewidth=2, alpha=0.4)
        ax2.set_title("Predicted Layout (HouseDiffusion)", fontsize=14, fontweight='bold')
        ax2.axis('equal')
        ax2.axis('off')
        
        # 3. Ground Truth Rooms
        gt_gdf = gpd.GeoDataFrame(
            geometry=[r[0] for r in gt_rooms],
            data={"roomtype": [r[1] for r in gt_rooms]}
        )
        gt_gdf.plot(ax=ax3, column="roomtype", legend=True, cmap='Set3', edgecolor='white', linewidth=1.5)
        gpd.GeoDataFrame(geometry=[outline_geom]).plot(ax=ax3, facecolor='none', edgecolor='black', linewidth=2, alpha=0.4)
        ax3.set_title("Ground Truth Layout", fontsize=14, fontweight='bold')
        ax3.axis('equal')
        ax3.axis('off')
        
        plt.suptitle(f"Layout Generation Comparison (Floor ID: {floor_id})", fontsize=16)
        plt.tight_layout()
        
        plot_path = os.path.join(output_dir, f"comparison_{floor_id}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Comparison plot saved successfully to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Outline-Conditioned HouseDiffusion Model")
    parser.add_argument("--csv_path", type=str, default="mds_V2_5.372k.csv", help="Path to room geometries CSV")
    parser.add_argument("--train_index", type=str, default="train_indices.txt", help="Path to train split index file")
    parser.add_argument("--test_index", type=str, default="test_indices.txt", help="Path to test split index file")
    parser.add_argument("--model_path", type=str, default="model_final.pth", help="Path to trained model weights")
    
    import sys
    if len(sys.argv) == 1:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()
        
    validate(args)
