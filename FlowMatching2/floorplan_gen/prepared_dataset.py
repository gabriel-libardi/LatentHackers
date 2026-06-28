from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import DEFAULT_VERTEX_COUNT
from .representations import legacy_tokens_to_vertices


class PreparedFloorPlanDataset(Dataset):
    def __init__(self, prepared_dir: str | Path, split: str = "train") -> None:
        self.prepared_dir = Path(prepared_dir)
        path = self.prepared_dir / f"{split}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        data = np.load(path, allow_pickle=True)
        self.plan_ids = data["plan_ids"].astype(str).tolist()
        self.boundary_points = data["boundary_points"].astype(np.float32)
        self.room_tokens = data["room_tokens"].astype(np.float32)
        self.room_masks = data["room_masks"].astype(bool)
        if "room_vertices" in data:
            self.room_vertices = data["room_vertices"].astype(np.float32)
        else:
            self.room_vertices = np.asarray(
                [legacy_tokens_to_vertices(tokens, DEFAULT_VERTEX_COUNT) for tokens in self.room_tokens],
                dtype=np.float32,
            )
        self.room_presence = (
            data["room_presence"].astype(np.float32)
            if "room_presence" in data
            else self.room_tokens[..., 0].astype(np.float32)
        )
        self.room_type_ids = (
            data["room_type_ids"].astype(np.int64)
            if "room_type_ids" in data
            else self.room_tokens[..., 1].astype(np.int64)
        )
        self.room_counts = self.room_masks.sum(axis=1).astype(np.int64)
        self.room_geometry = (
            data["room_geometry"].astype(np.float32)
            if "room_geometry" in data
            else self.room_vertices.reshape(self.room_vertices.shape[0], self.room_vertices.shape[1], -1)
        )
        self.wall_graph = {}
        for key in [
            "junction_xy",
            "junction_mask",
            "edge_index",
            "edge_mask",
            "edge_is_exterior",
            "edge_room_ids",
            "wall_room_types",
            "wall_room_mask",
        ]:
            if key in data:
                self.wall_graph[key] = data[key]
        self.metadata = json.loads((self.prepared_dir / "metadata.json").read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self.plan_ids)

    def __getitem__(self, index: int) -> dict[str, object]:
        item = {
            "plan_id": self.plan_ids[index],
            "boundary_points": torch.from_numpy(self.boundary_points[index]),
            "room_tokens": torch.from_numpy(self.room_tokens[index]),
            "room_geometry": torch.from_numpy(self.room_geometry[index]),
            "room_vertices": torch.from_numpy(self.room_vertices[index]),
            "room_presence": torch.from_numpy(self.room_presence[index]),
            "room_type_ids": torch.from_numpy(self.room_type_ids[index]),
            "room_mask": torch.from_numpy(self.room_masks[index]),
            "room_count": torch.tensor(self.room_counts[index], dtype=torch.long),
        }
        if self.wall_graph:
            item.update(
                {
                    "junction_xy": torch.from_numpy(self.wall_graph["junction_xy"][index].astype(np.float32)),
                    "junction_mask": torch.from_numpy(self.wall_graph["junction_mask"][index].astype(bool)),
                    "edge_index": torch.from_numpy(self.wall_graph["edge_index"][index].astype(np.int64)),
                    "edge_mask": torch.from_numpy(self.wall_graph["edge_mask"][index].astype(bool)),
                    "edge_is_exterior": torch.from_numpy(self.wall_graph["edge_is_exterior"][index].astype(bool)),
                    "edge_room_ids": torch.from_numpy(self.wall_graph["edge_room_ids"][index].astype(np.int64)),
                    "wall_room_types": torch.from_numpy(self.wall_graph["wall_room_types"][index].astype(np.int64)),
                    "wall_room_mask": torch.from_numpy(self.wall_graph["wall_room_mask"][index].astype(bool)),
                }
            )
        return item
