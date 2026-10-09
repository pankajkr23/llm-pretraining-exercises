"""The ledger counts what it is given, and the world knows which links leave a node.

No `torch` here. The ledger is plain arithmetic over byte counts, and a `Rank` only needs objects
that report `numel()` and `element_size()` — so a stand-in is used, and these guards run in the
ordinary CI job rather than only where the train extra is installed.
"""

import pytest
from zerosim.world import CATEGORIES, PERSISTENT, TRANSIENT, MemoryLedger, Rank, World


class _Blob:
    """Anything with a size, standing in for a tensor."""

    def __init__(self, numel: int, itemsize: int) -> None:
        self._numel, self._itemsize = numel, itemsize

    def numel(self) -> int:
        return self._numel

    def element_size(self) -> int:
        return self._itemsize


def test_the_categories_partition_into_persistent_and_transient() -> None:
    assert set(PERSISTENT).isdisjoint(TRANSIENT)
    assert set(CATEGORIES) == set(PERSISTENT) | set(TRANSIENT)


def test_the_peak_is_a_high_water_mark_not_the_current_value() -> None:
    ledger = MemoryLedger()
    ledger.allocate("grad_bucket", "a", 100)
    ledger.free("grad_bucket", "a", 100)
    ledger.allocate("grad_bucket", "b", 40)
    assert ledger.current["grad_bucket"] == 40
    assert ledger.peak["grad_bucket"] == 100


def test_the_total_peak_is_not_the_sum_of_category_peaks() -> None:
    """Two transients that never coexist must not be added together."""
    ledger = MemoryLedger()
    ledger.allocate("gathered_params", "u", 100)
    ledger.free("gathered_params", "u", 100)
    ledger.allocate("grad_bucket", "u", 100)
    ledger.free("grad_bucket", "u", 100)
    assert ledger.peak["gathered_params"] + ledger.peak["grad_bucket"] == 200
    assert ledger.peak_total == 100


def test_freeing_more_than_was_held_is_refused() -> None:
    ledger = MemoryLedger()
    ledger.allocate("params", "u", 10)
    with pytest.raises(RuntimeError, match="unbalanced"):
        ledger.free("params", "u", 11)


def test_an_unknown_category_is_refused() -> None:
    with pytest.raises(KeyError, match="unknown memory category"):
        MemoryLedger().allocate("activations", "x", 1)


def test_a_rank_charges_numel_times_itemsize_and_credits_it_back() -> None:
    rank = Rank(0, 0)
    rank.put("params", "u", _Blob(1000, 2))
    rank.put("master", "u", _Blob(1000, 4))
    assert rank.ledger.current["params"] == 2000
    assert rank.ledger.held == 6000
    rank.drop("params", "u")
    assert rank.ledger.current["params"] == 0
    assert not rank.holds("params", "u")
    assert rank.ledger.peak["params"] == 2000


def test_a_rank_refuses_to_overwrite_what_it_already_holds() -> None:
    """A silent overwrite would leave the old bytes charged with nothing holding them."""
    rank = Rank(0, 0)
    rank.put("grads", "u", _Blob(10, 2))
    with pytest.raises(KeyError, match="already holds"):
        rank.put("grads", "u", _Blob(10, 2))


def test_thirty_two_ranks_on_four_nodes_cross_a_node_boundary_four_times() -> None:
    world = World(32, 8)
    kinds = [kind for _, _, kind in world.ring_links()]
    assert kinds.count("inter") == 4
    assert kinds.count("intra") == 28
    inter = [(src, dst) for src, dst, kind in world.ring_links() if kind == "inter"]
    assert inter == [(7, 8), (15, 16), (23, 24), (31, 0)]


def test_one_node_has_no_inter_node_link() -> None:
    world = World(8)
    assert all(kind == "intra" for _, _, kind in world.ring_links())


def test_a_world_that_does_not_fill_its_nodes_is_refused() -> None:
    with pytest.raises(ValueError, match="do not fill"):
        World(10, 4)


def test_the_ring_visits_every_rank_once() -> None:
    world = World(12, 4)
    sources = [src for src, _, _ in world.ring_links()]
    targets = [dst for _, dst, _ in world.ring_links()]
    assert sorted(sources) == sorted(targets) == list(range(12))
