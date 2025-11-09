import sys
from pathlib import Path

import igraph as ig
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import motif_search as ms  # noqa: E402

FFL_SIGNATURES = {
    "Coherent Type 1 (C1)": (1.0, 1.0, 1.0),
    "Coherent Type 2 (C2)": (1.0, -1.0, -1.0),
    "Coherent Type 3 (C3)": (-1.0, 1.0, -1.0),
    "Coherent Type 4 (C4)": (-1.0, -1.0, 1.0),
    "Incoherent Type 1 (I1)": (1.0, 1.0, -1.0),
    "Incoherent Type 2 (I2)": (1.0, -1.0, 1.0),
    "Incoherent Type 3 (I3)": (-1.0, 1.0, 1.0),
    "Incoherent Type 4 (I4)": (-1.0, -1.0, -1.0),
}

@pytest.fixture
def coherent_ffl_adj():
    """Adjacency with one coherent type 1 feed-forward loop (0->1->2)."""
    return np.array(
        [
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=float,
    )


@pytest.fixture
def balancing_cycle_graph():
    """Graph with a 3-node balancing feedback loop (product of signs < 0)."""
    adj = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    g = ig.Graph.Weighted_Adjacency(adj.tolist(), mode="directed")
    g.vs["name"] = list(range(adj.shape[0]))
    return g


@pytest.fixture
def mixed_motif_adj():
    """Adjacency with one coherent FFL and one balancing 2-node feedback loop."""
    return np.array(
        [
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=float,
    )

# Matrix covering multiple motifs and feedback loops
@pytest.fixture
def comprehensive_adj():
    """
    Matrix with the following motifs:
        - C1, I1, and I2 FFLs
        - 2 balancing 2-node loops, 1 reinforcing 3-node loop, 1 reinforcing 4-node loop
    """
    return np.array(
        [
            [0.0, 1.0, 1.0, -1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )


@pytest.fixture
def double_ffl_adj():
    """Adjacency containing C2 and I4 feed-forward loops."""
    return np.array(
        [
            [0.0, 1.0, -1.0, 0.0],
            [0.0, 0.0, -1.0, -1.0],
            [0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )


@pytest.fixture
def feedback_only_adj():
    """Adjacency with only feedback loops of different sizes."""
    return np.array(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ],
        dtype=float,
    )


@pytest.fixture
def no_motif_adj():
    """Adjacency with no three-node motifs or feedback loops."""
    return np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )


@pytest.fixture
def larger_graph_adj():
    """Five-node adjacency with mixed coherent motif types."""
    return np.array(
        [
            [0.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )


@pytest.fixture
def disconnected_pairs_adj():
    """Adjacency with two disjoint dyads used to test subgraph counting."""
    return np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )


@pytest.fixture
def zero_weight_ffl_graph():
    """igraph graph whose FFL includes an edge of weight zero."""
    g = ig.Graph(directed=True)
    g.add_vertices(3)
    g.add_edges([(0, 1), (0, 2), (1, 2)])
    g.es["weight"] = [0.0, 1.0, -1.0]
    return g

# Assert statements

def test_classify_ffl_coherent_type1(coherent_ffl_adj):
    g = ig.Graph.Weighted_Adjacency(coherent_ffl_adj.tolist(), mode="directed")
    assert ms.classify_ffl(g, 0, 1, 2) == "Coherent Type 1 (C1)"


@pytest.mark.parametrize("ffl_type, signs", FFL_SIGNATURES.items())
def test_classify_ffl_all_types(ffl_type, signs):
    adj = np.zeros((3, 3), dtype=float)
    adj[0, 1], adj[1, 2], adj[0, 2] = signs
    g = ig.Graph.Weighted_Adjacency(adj.tolist(), mode="directed")
    assert ms.classify_ffl(g, 0, 1, 2) == ffl_type

def test_count_ffl_types_detects_single_loop(coherent_ffl_adj):
    """Adjacency with one coherent type 1 feed-forward loop (0->1->2)."""
    details, counts = ms.count_ffl_types(coherent_ffl_adj)
    assert counts["Coherent Type 1 (C1)"] == 1
    assert sum(counts.values()) == 1
    assert details and details[0]["FFL_Type"] == "Coherent Type 1 (C1)"


def test_classify_feedback_loop_balancing(balancing_cycle_graph):
    assert (
        ms.classify_feedback_loop(balancing_cycle_graph, [0, 1, 2])
        == "Balancing Feedback"
    )


def test_count_feedback_loops_limits_size(mixed_motif_adj):
    loops, counts = ms.count_feedback_loops(mixed_motif_adj, max_size=3)
    assert 2 in counts
    assert counts[2]["Balancing Feedback"] == 1
    assert loops[2]["Balancing Feedback"][0] == [2, 3]


def test_calculate_motif_concentration_uses_connected_subgraphs(coherent_ffl_adj):
    motif_counts = {"Coherent Type 1 (C1)": 1, "Incoherent Type 1 (I1)": 0}
    concentrations = ms.calculate_motif_concentration(coherent_ffl_adj, motif_counts, 3)
    assert concentrations["Coherent Type 1 (C1)"] == pytest.approx(1.0)
    assert concentrations["Incoherent Type 1 (I1)"] == pytest.approx(0.0)


def test_analyze_network_motifs_summary(mixed_motif_adj):
    results = ms.analyze_network_motifs(mixed_motif_adj, max_fbl_size=3)
    summary = results["summary"]
    assert summary["total_ffls"] == 1
    assert summary["total_fbls"] == 1
    assert summary["fbl_proportion"]["balancing"] == 1
    assert summary["fbl_proportion"]["reinforcing"] == 0


def test_comprehensive_counts(comprehensive_adj):
    _, ffl_counts = ms.count_ffl_types(comprehensive_adj)
    assert ffl_counts["Coherent Type 1 (C1)"] == 1
    assert ffl_counts["Incoherent Type 1 (I1)"] == 1
    assert ffl_counts["Incoherent Type 2 (I2)"] == 1

    concentrations = ms.calculate_motif_concentration(comprehensive_adj, ffl_counts, 3)
    assert concentrations["Coherent Type 1 (C1)"] == pytest.approx(0.25)
    assert concentrations["Incoherent Type 1 (I1)"] == pytest.approx(0.25)
    assert concentrations["Incoherent Type 2 (I2)"] == pytest.approx(0.25)
    for motif, value in concentrations.items():
        if motif not in {"Coherent Type 1 (C1)", "Incoherent Type 1 (I1)", "Incoherent Type 2 (I2)"}:
            assert value == pytest.approx(0.0)

    loops, counts = ms.count_feedback_loops(comprehensive_adj, max_size=4)
    assert counts[2]["Balancing Feedback"] == 2
    assert counts[3]["Reinforcing Feedback"] == 1
    assert counts[4]["Reinforcing Feedback"] == 1
    assert loops[2]["Balancing Feedback"] == [[0, 3], [1, 2]]
    assert loops[3]["Reinforcing Feedback"] == [[0, 2, 3]]
    assert loops[4]["Reinforcing Feedback"] == [[0, 1, 2, 3]]

    # Feedback loop concentrations
    conc_size2 = ms.calculate_motif_concentration(comprehensive_adj, counts[2], 2)
    conc_size3 = ms.calculate_motif_concentration(comprehensive_adj, counts[3], 3)
    conc_size4 = ms.calculate_motif_concentration(comprehensive_adj, counts[4], 4)
    assert conc_size2["Balancing Feedback"] == pytest.approx(0.4)
    assert conc_size3["Reinforcing Feedback"] == pytest.approx(0.25)
    assert conc_size4["Reinforcing Feedback"] == pytest.approx(1.0)


def test_double_ffl_counts(double_ffl_adj):
    _, counts = ms.count_ffl_types(double_ffl_adj)
    assert counts["Coherent Type 2 (C2)"] == 1
    assert counts["Incoherent Type 4 (I4)"] == 1
    for motif, count in counts.items():
        if motif not in {"Coherent Type 2 (C2)", "Incoherent Type 4 (I4)"}:
            assert count == 0


def test_feedback_only_loops(feedback_only_adj):
    _, counts = ms.count_feedback_loops(feedback_only_adj, max_size=4)
    assert counts[2]["Balancing Feedback"] == 1
    assert counts[3]["Reinforcing Feedback"] == 1
    assert 4 not in counts


def test_no_motif_matrix(no_motif_adj):
    _, ffl_counts = ms.count_ffl_types(no_motif_adj)
    assert sum(ffl_counts.values()) == 0

    _, loop_counts = ms.count_feedback_loops(no_motif_adj, max_size=4)
    assert loop_counts == {}


def test_zero_weight_edge_ffl(zero_weight_ffl_graph):
    assert ms.classify_ffl(zero_weight_ffl_graph, 0, 1, 2) == "Unknown"


def test_larger_graph_motifs(larger_graph_adj):
    _, counts = ms.count_ffl_types(larger_graph_adj)
    assert counts["Coherent Type 1 (C1)"] == 1
    assert counts["Coherent Type 4 (C4)"] == 1
    assert sum(counts.values()) == 2


def test_concentration_skips_disconnected_subgraphs(disconnected_pairs_adj):
    motif_counts = {"Dummy Motif": 2}
    conc = ms.calculate_motif_concentration(disconnected_pairs_adj, motif_counts, 2)
    assert conc["Dummy Motif"] == pytest.approx(1.0)


def test_visualize_flag_calls_helper(monkeypatch, coherent_ffl_adj):
    called = {"visualize": False}

    def fake_visualize(adj, phen):
        called["visualize"] = True
        assert phen == [1, 1, 1]

    monkeypatch.setattr(ms, "visualize_graph", fake_visualize)
    ms.count_ffl_types(coherent_ffl_adj, visualize=True, phen_opt=[1, 1, 1])
    assert called["visualize"]
