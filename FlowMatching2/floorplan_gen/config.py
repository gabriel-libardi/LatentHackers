"""Shared defaults for the MSD floor-plan baseline."""

CSV_PATH = "data/modified-swiss-dwellings/mds_V2_5.372k.csv"
ORIGINAL_SPLIT_DIR = "data/modified-swiss-dwellings/modified-swiss-dwellings-v2"

PLAN_ID_COL = "plan_id"
FLOOR_ID_COL = "floor_id"
ENTITY_TYPE_COL = "entity_type"
ROOM_ENTITY_TYPE = "area"
GEOM_COL = "geom"
ROOM_TYPE_COL = "roomtype"

OUTLINE_BUFFER_METERS = 0.3
DEFAULT_BOUNDARY_POINTS = 256
DEFAULT_MAX_ROOMS = 160
DEFAULT_VERTEX_COUNT = 16
PAD_TYPE_ID = 0
DEFAULT_SEED = 42
