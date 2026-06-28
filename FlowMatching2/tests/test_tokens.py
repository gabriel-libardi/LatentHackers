import pytest

np = pytest.importorskip("numpy")
shapely_wkt = pytest.importorskip("shapely.wkt")

from floorplan_gen.tokens import RoomRecord, build_room_type_vocab, make_room_tokens


def test_make_room_tokens_pads_and_sets_mask():
    room = RoomRecord(
        room_type="Bedroom",
        geometry=shapely_wkt.loads("POLYGON ((0 0, 2 0, 2 1, 0 1, 0 0))"),
    )
    vocab = build_room_type_vocab(["Bedroom"])

    tokens, mask = make_room_tokens([room], vocab, max_rooms=3)

    assert tokens.shape == (3, 6)
    assert mask.tolist() == [True, False, False]
    np.testing.assert_allclose(tokens[0], [1.0, 1.0, 1.0, 0.5, 2.0, 1.0])
    np.testing.assert_allclose(tokens[1:], 0.0)

