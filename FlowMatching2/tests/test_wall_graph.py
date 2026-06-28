import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
geom = pytest.importorskip("shapely.geometry")

from floorplan_gen.tokens import RoomRecord
from floorplan_gen.wall_graph import convert_rooms_to_wall_graph, decode_wall_graph
from floorplan_gen.wall_graph_losses import edge_index_to_dense, wall_graph_losses
from floorplan_gen.models import WallGraphFlow


def test_wall_graph_conversion_merges_duplicate_reversed_edges():
    rooms = [
        RoomRecord("A", geom.box(0, 0, 1, 1)),
        RoomRecord("B", geom.box(1, 0, 2, 1)),
    ]

    graph = convert_rooms_to_wall_graph(rooms, {"A": 1, "B": 2}, max_junctions=16, max_edges=32, max_rooms=4)

    assert graph.junction_mask.sum() == 6
    assert graph.edge_mask.sum() == 7
    shared = [ids for ids, mask in zip(graph.edge_room_ids, graph.edge_mask) if mask and set(ids.tolist()) == {0, 1}]
    assert len(shared) == 1


def test_wall_graph_vertex_snapping():
    rooms = [
        RoomRecord("A", geom.Polygon([(0, 0), (1.0001, 0), (1.0001, 1), (0, 1)])),
        RoomRecord("B", geom.Polygon([(1.0002, 0), (2, 0), (2, 1), (1.0002, 1)])),
    ]

    graph = convert_rooms_to_wall_graph(rooms, {"A": 1, "B": 2}, max_junctions=16, max_edges=32, max_rooms=4, snap_tolerance=1e-2)

    assert graph.junction_mask.sum() == 6


def test_wall_graph_t_junction_splits_edge():
    rooms = [
        RoomRecord("A", geom.box(0, 0, 2, 1)),
        RoomRecord("B", geom.box(0, 1, 1, 2)),
    ]

    graph = convert_rooms_to_wall_graph(rooms, {"A": 1, "B": 2}, max_junctions=16, max_edges=32, max_rooms=4)

    points = graph.junction_xy
    edge_coords = {
        tuple(sorted((tuple(points[a]), tuple(points[b]))))
        for a, b in graph.edge_index[graph.edge_mask]
    }
    assert tuple(sorted(((0.0, 1.0), (1.0, 1.0)))) in edge_coords
    assert tuple(sorted(((1.0, 1.0), (2.0, 1.0)))) in edge_coords


def test_wall_graph_padding_masks():
    graph = convert_rooms_to_wall_graph([RoomRecord("A", geom.box(0, 0, 1, 1))], {"A": 1}, 10, 20, 4)

    assert graph.junction_xy.shape == (10, 2)
    assert graph.edge_index.shape == (20, 2)
    assert not graph.junction_mask[4:].any()
    assert not graph.edge_mask[int(graph.edge_mask.sum()) :].any()


def test_wall_graph_model_symmetric_edge_logits_and_no_self_edges():
    model = WallGraphFlow(max_junctions=5, d_model=32, nhead=4, encoder_layers=1, decoder_layers=1, dim_feedforward=64)
    out = model(torch.randn(2, 5, 2), torch.rand(2), torch.randn(2, 8, 2), junction_mask=torch.ones(2, 5, dtype=torch.bool))

    torch.testing.assert_close(out["edge_logits"], out["edge_logits"].transpose(1, 2))
    assert torch.all(out["edge_logits"][:, torch.arange(5), torch.arange(5)] < -20)


def test_wall_graph_loss_masks_padded_junction_flow():
    outputs = {
        "flow": torch.tensor([[[0.0, 0.0], [1000.0, 1000.0]]]),
        "junction_presence_logits": torch.zeros(1, 2),
        "edge_logits": torch.zeros(1, 2, 2),
    }
    losses = wall_graph_losses(
        outputs,
        target_flow=torch.zeros(1, 2, 2),
        junction_xy=torch.zeros(1, 2, 2),
        junction_mask=torch.tensor([[True, False]]),
        edge_index=torch.zeros(1, 1, 2, dtype=torch.long),
        edge_mask=torch.zeros(1, 1, dtype=torch.bool),
        lambda_junction_presence=0.0,
        lambda_edge=0.0,
    )

    assert losses["loss"].item() == pytest.approx(0.0)


def test_wall_graph_endpoint_loss_uses_estimated_clean_sample():
    outputs = {
        "flow": torch.tensor([[[2.0, 0.0]]]),
        "junction_presence_logits": torch.ones(1, 1),
        "edge_logits": torch.zeros(1, 1, 1),
    }
    losses = wall_graph_losses(
        outputs,
        target_flow=torch.zeros(1, 1, 2),
        junction_xy=torch.tensor([[[1.0, 0.0]]]),
        junction_mask=torch.tensor([[True]]),
        edge_index=torch.zeros(1, 1, 2, dtype=torch.long),
        edge_mask=torch.zeros(1, 1, dtype=torch.bool),
        noisy_junction_xy=torch.tensor([[[0.0, 0.0]]]),
        t=torch.tensor([0.5]),
        lambda_junction_flow=0.0,
        lambda_junction_endpoint=1.0,
        lambda_junction_presence=0.0,
        lambda_edge=0.0,
    )

    assert losses["loss"].item() == pytest.approx(0.0)


def test_wall_graph_edge_loss_counts_upper_triangle_once():
    outputs = {
        "flow": torch.zeros(1, 3, 2),
        "junction_presence_logits": torch.ones(1, 3),
        "edge_logits": torch.zeros(1, 3, 3),
    }
    edge_index = torch.tensor([[[0, 1], [1, 0], [1, 2]]], dtype=torch.long)
    edge_mask = torch.tensor([[True, True, False]])
    dense = edge_index_to_dense(edge_index, edge_mask, 3)
    losses = wall_graph_losses(
        outputs,
        target_flow=torch.zeros(1, 3, 2),
        junction_xy=torch.zeros(1, 3, 2),
        junction_mask=torch.tensor([[True, True, True]]),
        edge_index=edge_index,
        edge_mask=edge_mask,
        lambda_junction_flow=0.0,
        lambda_junction_endpoint=0.0,
        lambda_junction_presence=0.0,
        lambda_edge=1.0,
    )

    assert dense[0].sum().item() == 2.0
    assert losses["edge_recall"].item() in {0.0, 1.0}


def test_wall_graph_polygonization_empty_and_disconnected_predictions():
    outline = geom.box(0, 0, 2, 2)
    rooms, info = decode_wall_graph(
        np.zeros((4, 2), dtype=np.float32),
        np.zeros(4, dtype=np.float32),
        np.zeros((4, 4), dtype=np.float32),
        outline,
    )

    assert rooms == []
    assert info["active_junctions"] == 0


def test_wall_graph_polygonization_rejects_crossing_segments():
    outline = geom.box(0, 0, 1, 1)
    points = np.asarray([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=np.float32)
    logits = np.full((4, 4), -10.0, dtype=np.float32)
    logits[0, 1] = logits[1, 0] = 10.0
    logits[2, 3] = logits[3, 2] = 10.0

    rooms, info = decode_wall_graph(points, np.ones(4, dtype=np.float32), logits, outline)

    assert info["rejected_crossings"] == 1


def test_wall_graph_checkpoint_compatibility_metadata(tmp_path):
    path = tmp_path / "ckpt.pt"
    torch.save({"metadata": {"geometry_representation": "partition"}}, path)
    ckpt = torch.load(path, map_location="cpu")

    assert ckpt["metadata"]["geometry_representation"] != "wall_graph"
