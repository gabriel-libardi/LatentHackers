from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from .config import PAD_TYPE_ID


@dataclass(frozen=True)
class RoomRecord:
    room_type: str
    geometry: object


def build_room_type_vocab(room_types: Iterable[str]) -> dict[str, int]:
    """Build a stable room vocabulary, reserving 0 for padding."""

    labels = sorted({str(label) for label in room_types if str(label)})
    return {label: index + 1 for index, label in enumerate(labels)}


def invert_vocab(vocab: Mapping[str, int]) -> dict[int, str]:
    return {int(value): key for key, value in vocab.items()}


def room_box_features(geometry) -> tuple[float, float, float, float]:
    """Return centroid x/y and bounding-box width/height for one room."""

    min_x, min_y, max_x, max_y = geometry.bounds
    centroid = geometry.centroid
    return (
        float(centroid.x),
        float(centroid.y),
        float(max_x - min_x),
        float(max_y - min_y),
    )


def make_room_tokens(
    rooms: Iterable[RoomRecord],
    type_to_id: Mapping[str, int],
    max_rooms: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create padded room tokens.

    Token columns are: presence, type_id, centroid_x, centroid_y, width, height.
    """

    if max_rooms <= 0:
        raise ValueError("max_rooms must be positive.")

    tokens = np.zeros((max_rooms, 6), dtype=np.float32)
    mask = np.zeros((max_rooms,), dtype=bool)

    sorted_rooms = sorted(
        list(rooms),
        key=lambda room: (room.room_type, room.geometry.centroid.x, room.geometry.centroid.y),
    )
    for index, room in enumerate(sorted_rooms[:max_rooms]):
        cx, cy, width, height = room_box_features(room.geometry)
        tokens[index] = np.asarray(
            [
                1.0,
                float(type_to_id.get(room.room_type, PAD_TYPE_ID)),
                cx,
                cy,
                width,
                height,
            ],
            dtype=np.float32,
        )
        mask[index] = True
    return tokens, mask

