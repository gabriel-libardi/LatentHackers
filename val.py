import argparse
import os
import torch
import pandas as pd
import gpd = None
try:
    import geopandas as gpd
except ImportError:
    pass
from shapely import wkt
import shapely.affinity
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
import numpy as np

from dataset import FloorplanDataset, ROOM_TYPES, ROOM_MAPPING
from house_diffusion import HouseDiffusionModel, GaussianDiffusion, RoomConditionSampler, generate_floorplan

class OutlineConditionedGenerator:
    """
    Generates the room partitions for a given bare outline by sampling room topologies
    probabilistically from dataset histograms, using the trained HouseDiffusion model.
    """
    def __init__(self, model_path, d_model=256, num_layers=6, device="cpu"):
        self.device = torch.device(device)
        self.max_outline_len = 128
        
        # Initialize condition sampler using standard training histograms
        self.condition_sampler = RoomConditionSampler(max_total_corners=128)
        
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
        
    def generate(self, outline_polygon):
        """
        Main entry point for generating room partitions given a bare apartment outline.
        Uses the generate_floorplan method from house_diffusion to sample layouts.
        Returns:
            List of tuples: (shapely.geometry.Polygon, room_type_name)
        """
        # 1. Normalize outline polygon coordinates
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
        
        # 2. Use generate_floorplan to run reverse denoising
        print("Denoising coordinates with generate_floorplan...")
        layouts = generate_floorplan(
            model=self.model,
            diffusion=self.diffusion,
            condition_sampler=self.condition_sampler,
            outline=outline_tensor,
            outline_mask=outline_mask,
            max_corners=128
        )
        
        layout = layouts[0]
        sampled_normalized = layout["coordinates"].cpu().numpy()  # (N_active, 2)
        entity_type_np = layout["room_types"].cpu().numpy()       # (N_active,)
        entity_idx_np = layout["room_ids"].cpu().numpy()          # (N_active,)
        
        # 3. De-normalize generated coordinates back to original scale
        generated_corners = []
        for cx, cy in sampled_normalized:
            gx = cx * max_size + minx
            gy = cy * max_size + miny
            generated_corners.append((gx, gy))
            
        # 4. Reconstruct room polygons
        rooms = {}
        for i in range(len(sampled_normalized)):
            ent_id = entity_idx_np[i]
            if ent_id not in rooms:
                rooms[ent_id] = {
                    "coords": [],
                    "type_idx": entity_type_np[i]
                }
            rooms[ent_id]["coords"].append(generated_corners[i])
            
        generated_polygons = []
        for ent_id, room in rooms.items():
            if len(room["coords"]) >= 3:
                try:
                    poly = Polygon(room["coords"])
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
        if pred_rooms and gpd is not None:
            pred_gdf = gpd.GeoDataFrame(
                geometry=[r[0] for r in pred_rooms],
                data={"roomtype": [r[1] for r in pred_rooms]}
            )
            pred_gdf.plot(ax=ax2, column="roomtype", legend=True, cmap='Set3', edgecolor='white', linewidth=1.5)
        else:
            print("Warning: No rooms generated for outline.")
        if gpd is not None:
            gpd.GeoDataFrame(geometry=[outline_geom]).plot(ax=ax2, facecolor='none', edgecolor='black', linewidth=2, alpha=0.4)
        ax2.set_title("Predicted Layout (HouseDiffusion)", fontsize=14, fontweight='bold')
        ax2.axis('equal')
        ax2.axis('off')
        
        # 3. Ground Truth Rooms
        if gpd is not None:
            gt_gdf = gpd.GeoDataFrame(
                geometry=[gt_rooms[i][0] for i in range(len(gt_rooms))],
                data={"roomtype": [gt_rooms[i][1] for i in range(len(gt_rooms))]}
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
    parser.add_argument("--test_index", type=str, default="test_indices.txt", help="Path to test split index file")
    parser.add_argument("--model_path", type=str, default="model_final.pth", help="Path to trained model weights")
    
    import sys
    if len(sys.argv) == 1:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()
        
    validate(args)
