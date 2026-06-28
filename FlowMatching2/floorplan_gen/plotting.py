ROOM_TYPE_COLORS = {
    "Balcony": "#59a14f",
    "Bathroom": "#4e79a7",
    "Bedroom": "#f28e2b",
    "Corridor": "#edc948",
    "Dining": "#b07aa1",
    "Kitchen": "#e15759",
    "Livingroom": "#76b7b2",
    "Stairs": "#9c755f",
    "Storeroom": "#bab0ac",
    "Structure": "#8cd17d",
}


def room_color(room_type: str) -> str:
    return ROOM_TYPE_COLORS.get(str(room_type), "#bdbdbd")
