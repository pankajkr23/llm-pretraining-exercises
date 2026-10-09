"""Flat buffers: how a unit's parameters become one 1-D tensor that can be cut into N equal shards.

**ZeRO does not shard a model by layers.** It shards *each* flat buffer across *all* ranks: rank `r`
owns the `r`-th of `N` equal slices of every unit's buffer. That is the difference between ZeRO and
pipeline parallelism, and it is easy to get backwards — a ZeRO-3 rank does not "hold layers 4–7",
it holds 1/N of every layer, and gathers the rest of a layer only for the moment it is computed.

**A unit** is the piece ZeRO-3 gathers at once: here the embeddings, each transformer block, and the
final norm with the output head. Its parameters are concatenated in a fixed order into one buffer
— the layout this module records — and the same layout is used by every stage, for the weights,
the gradients, the master copy and both Adam moments. Using one partitioning everywhere is what
makes the stages comparable: the only thing that changes between them is what each rank *keeps*.

**Padding.** A buffer of `numel` elements is padded with zeros to the next multiple of `N` so the
shards are equal, which the ring collectives require. Padding is real memory and real traffic, so
it is counted rather than hidden — `padded_numel` is what every byte count in this exercise is
computed from, and `Layout.padding` says how much of it is padding. A zero weight with a zero
gradient stays exactly zero under AdamW (weight decay of zero is zero, and the update is
`0 / (0 + eps)`), so padding never leaks into the model.
"""

from dataclasses import dataclass

import torch


def padded_numel(numel: int, world_size: int) -> int:
    """The smallest multiple of `world_size` that is at least `numel`."""
    return -(-numel // world_size) * world_size


@dataclass(frozen=True)
class Entry:
    """One parameter's place inside a unit's flat buffer."""

    name: str
    shape: tuple[int, ...]
    offset: int
    numel: int


@dataclass(frozen=True)
class Layout:
    """A unit's parameters, in order, and the padded flat buffer they become at `world_size`.

    Attributes:
        unit: The unit's name, e.g. `"block0"`.
        entries: Each parameter's name, shape and offset.
        numel: Real parameters in the unit.
        world_size: The `N` the padding was computed for.
    """

    unit: str
    entries: tuple[Entry, ...]
    numel: int
    world_size: int

    @property
    def padded(self) -> int:
        """Buffer length after padding to a multiple of `world_size`."""
        return padded_numel(self.numel, self.world_size)

    @property
    def padding(self) -> int:
        """Zero elements appended so the buffer divides into equal shards."""
        return self.padded - self.numel

    @property
    def shard(self) -> int:
        """Elements in one rank's shard."""
        return self.padded // self.world_size


def layout(unit: str, named_shapes: list[tuple[str, tuple[int, ...]]], world_size: int) -> Layout:
    """Lay `named_shapes` out back to back, in the order given."""
    entries, offset = [], 0
    for name, shape in named_shapes:
        numel = 1
        for dim in shape:
            numel *= dim
        entries.append(Entry(name, tuple(shape), offset, numel))
        offset += numel
    return Layout(unit, tuple(entries), offset, world_size)


def flatten(tensors: dict[str, torch.Tensor], spec: Layout, dtype: torch.dtype) -> torch.Tensor:
    """Concatenate the unit's parameters into one zero-padded buffer of `spec.padded` elements."""
    flat = torch.zeros(spec.padded, dtype=dtype)
    for entry in spec.entries:
        flat[entry.offset : entry.offset + entry.numel].copy_(tensors[entry.name].reshape(-1))
    return flat


def views(flat: torch.Tensor, spec: Layout) -> dict[str, torch.Tensor]:
    """Shaped views into `flat`, one per parameter — no copy, so a gradient flows into `flat`."""
    return {
        entry.name: flat[entry.offset : entry.offset + entry.numel].view(entry.shape)
        for entry in spec.entries
    }


def shard_of(flat: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Rank `rank`'s contiguous slice of a padded flat buffer — a view, not a copy."""
    size = flat.numel() // world_size
    return flat[rank * size : (rank + 1) * size]
