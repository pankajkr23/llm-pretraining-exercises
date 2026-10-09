"""The ring collectives compute the right answer and send exactly the bytes the formulas say.

**Integer-valued floats, on purpose.** A ring adds the ranks' values in a different order from
`torch.stack(...).sum(0)`, so on arbitrary floats the two differ in the last bit and an exact
comparison would fail for a reason unrelated to correctness. Small integers are represented and
summed exactly in fp32 whatever the order, so these tests can demand **exact** equality — which is
the only strength at which "the ring computes the sum" is worth asserting.
"""

import pytest

torch = pytest.importorskip("torch")

from zerosim import flat  # noqa: E402
from zerosim.collectives import all_gather, all_reduce, reduce_scatter  # noqa: E402
from zerosim.world import World  # noqa: E402

SIZES = [2, 3, 8, 32]


def _buffers(n: int, numel: int, dtype=torch.float32, seed: int = 0) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.randint(-50, 50, (numel,), generator=generator).to(dtype) for _ in range(n)]


def _padded(raw: list[torch.Tensor], n: int) -> list[torch.Tensor]:
    """Zero-pad to a multiple of n, the way flat.py does."""
    size = flat.padded_numel(raw[0].numel(), n)
    out = []
    for tensor in raw:
        padded = torch.zeros(size, dtype=tensor.dtype)
        padded[: tensor.numel()] = tensor
        out.append(padded)
    return out


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("numel", [96, 97, 1000, 12_704])
def test_reduce_scatter_gives_rank_r_exactly_chunk_r_of_the_sum(n: int, numel: int) -> None:
    raw = _buffers(n, numel)
    buffers = _padded(raw, n)
    expected = torch.stack(buffers).sum(0).view(n, -1)
    shards = reduce_scatter(World(n), buffers)
    for rank in range(n):
        assert torch.equal(shards[rank], expected[rank]), f"rank {rank} holds the wrong chunk"
    # Padding is all zeros, so the real part of the reassembled sum is the naive sum unpadded.
    assert torch.equal(torch.cat(shards)[:numel], torch.stack(raw).sum(0))


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("numel", [96, 97, 1000])
def test_all_gather_gives_every_rank_every_shard_in_rank_order(n: int, numel: int) -> None:
    padded = flat.padded_numel(numel, n)
    shards = _buffers(n, padded // n, seed=1)
    gathered = all_gather(World(n), shards)
    expected = torch.cat(shards)
    for rank in range(n):
        assert torch.equal(gathered[rank], expected)


@pytest.mark.parametrize("n", SIZES)
def test_all_reduce_gives_every_rank_the_whole_sum(n: int) -> None:
    buffers = _padded(_buffers(n, 1001, seed=2), n)
    expected = torch.stack(buffers).sum(0)
    all_reduce(World(n), buffers)
    for buffer in buffers:
        assert torch.equal(buffer, expected)


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("numel", [97, 12_704])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_byte_counts_equal_p_times_n_minus_one_over_n_exactly(n: int, numel: int, dtype) -> None:
    """Every rank sends (N−1) chunks of padded/N elements: P·(N−1)/N per pass, to the byte."""
    padded = flat.padded_numel(numel, n)
    itemsize = torch.empty((), dtype=dtype).element_size()
    one_pass = padded * itemsize * (n - 1) // n
    assert padded * itemsize * (n - 1) % n == 0, "padding exists to make this exact"

    world = World(n)
    reduce_scatter(world, _padded(_buffers(n, numel, dtype), n))
    assert world.comm.sent == [one_pass] * n
    assert world.comm.received == [one_pass] * n

    world = World(n)
    all_gather(world, _buffers(n, padded // n, dtype))
    assert world.comm.sent == [one_pass] * n

    world = World(n)
    all_reduce(world, _padded(_buffers(n, numel, dtype), n))
    assert world.comm.sent == [2 * one_pass] * n
    assert world.comm.sent_by_op["reduce_scatter"] == [one_pass] * n
    assert world.comm.sent_by_op["all_gather"] == [one_pass] * n


def test_a_ring_link_only_ever_joins_neighbours() -> None:
    world = World(8, 4)
    all_reduce(world, _padded(_buffers(8, 80), 8))
    assert set(world.comm.link_bytes) == {(r, (r + 1) % 8) for r in range(8)}


def test_the_busiest_inter_node_link_is_recorded_per_call() -> None:
    """32 ranks on 4 nodes: each pass puts the same bytes on every link, inter-node included."""
    world = World(32, 8)
    numel = 32 * 10
    reduce_scatter(world, _buffers(32, numel))
    call = world.comm.calls[-1]
    per_link = numel * 4 * 31 // 32
    assert call.max_link_bytes == {"intra": per_link, "inter": per_link}


def test_reduce_scatter_does_not_modify_its_inputs() -> None:
    buffers = _buffers(4, 40)
    before = [b.clone() for b in buffers]
    reduce_scatter(World(4), buffers)
    assert all(torch.equal(a, b) for a, b in zip(before, buffers, strict=True))


def test_all_reduce_is_bit_identical_to_reduce_scatter_then_all_gather() -> None:
    """On real floats, not integers — this is the identity the stages lean on."""
    generator = torch.Generator().manual_seed(3)
    buffers = [torch.randn(64, generator=generator) for _ in range(8)]
    shards = reduce_scatter(World(8), [b.clone() for b in buffers])
    two_phase = all_gather(World(8), shards)
    one_call = all_reduce(World(8), [b.clone() for b in buffers])
    assert all(torch.equal(a, b) for a, b in zip(two_phase, one_call, strict=True))


def test_a_buffer_that_does_not_divide_is_refused_rather_than_padded_silently() -> None:
    with pytest.raises(ValueError, match="cannot be cut into 3 equal chunks"):
        reduce_scatter(World(3), _buffers(3, 10))


def test_one_buffer_per_rank_is_required() -> None:
    with pytest.raises(ValueError, match="one per rank"):
        reduce_scatter(World(4), _buffers(3, 12))


def test_a_single_rank_sends_nothing_and_keeps_its_own_buffer() -> None:
    world = World(1)
    buffers = _buffers(1, 10)
    shards = reduce_scatter(world, buffers)
    assert torch.equal(shards[0], buffers[0])
    assert world.comm.sent == [0]
