"""Thirty-two simulated devices in one process, each with its own memory and a ledger of it.

**A `Rank` is a device.** It owns its tensors, filed under a *category* — the weights, the
gradients, the fp32 master copy, the two Adam moments, and the transient buffers ZeRO creates and
frees. Every `put` and `drop` goes through the rank's `MemoryLedger`, which records the bytes of
the **storage** behind each tensor — not of the view — and keeps the high-water mark per category
and in total.

**The ledger measures; it never computes.** It has no idea what ZeRO is or what a formula says a
stage should hold. It adds up the byte sizes of the tensors the stages actually allocate. That is
the point: `formulas.py` predicts, the ledger measures, and the tests assert the two agree — which
they could not do if the ledger were written from the formula.

**A `World` is the cluster.** It owns the ranks, knows which node each sits on, and classifies every
ring link as intra-node or inter-node. Ranks are laid out in order — with 32 ranks on 4 nodes, ranks
0–7 are node 0 — and the ring visits them in that order, so exactly one link per node boundary (plus
the wrap-around) crosses between machines.

What is **not** recorded: activations. They are outside what ZeRO shards and are excluded from every
number this exercise reports. See `DECISIONS.md`.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-time only
    import torch

PERSISTENT = ("params", "grads", "master", "adam_m", "adam_v")
"""Categories a stage keeps for the whole run. The bytes-per-weight formulas describe these."""

TRANSIENT = ("gathered_params", "grad_bucket")
"""Categories that exist only for part of a step.

`gathered_params` is one unit's full weights, all-gathered by ZeRO-3 for its forward or backward
and freed straight after. `grad_bucket` is one unit's full gradient, produced by the backward pass
and freed as soon as it has been reduce-scattered (ZeRO-2 and ZeRO-3).
"""

CATEGORIES = PERSISTENT + TRANSIENT


@dataclass
class MemoryLedger:
    """Current and peak bytes per category, from the real size of every tensor put and dropped.

    Attributes:
        current: Bytes held now, per category.
        peak: The most ever held at once, per category.
        peak_total: The most ever held at once across all categories together. Not the sum of the
            per-category peaks — transient buffers peak at different moments.
        events: Every allocation and free, in order, as `(op, category, name, bytes)`.
    """

    current: dict[str, int] = field(default_factory=lambda: dict.fromkeys(CATEGORIES, 0))
    peak: dict[str, int] = field(default_factory=lambda: dict.fromkeys(CATEGORIES, 0))
    peak_total: int = 0
    events: list[tuple[str, str, str, int]] = field(default_factory=list)

    def allocate(self, category: str, name: str, nbytes: int) -> None:
        """Record `nbytes` arriving in `category`."""
        if category not in self.current:
            raise KeyError(f"unknown memory category {category!r}; expected one of {CATEGORIES}")
        self.current[category] += nbytes
        self.peak[category] = max(self.peak[category], self.current[category])
        self.peak_total = max(self.peak_total, sum(self.current.values()))
        self.events.append(("alloc", category, name, nbytes))

    def free(self, category: str, name: str, nbytes: int) -> None:
        """Record `nbytes` leaving `category`."""
        self.current[category] -= nbytes
        if self.current[category] < 0:
            raise RuntimeError(f"{category} went negative freeing {name}; the ledger is unbalanced")
        self.events.append(("free", category, name, nbytes))

    @property
    def held(self) -> int:
        """Bytes held now, all categories."""
        return sum(self.current.values())


@dataclass
class CallRecord:
    """One collective call: what it was, how big, and the bytes it put on each kind of link.

    Attributes:
        op: `"reduce_scatter"` or `"all_gather"`. An all-reduce is recorded as its two phases.
        label: What the call was for, e.g. `"grads/block0"`.
        numel: Elements in the full buffer on each rank.
        itemsize: Bytes per element.
        max_link_bytes: For each link kind (`"intra"`, `"inter"`), the most bytes any single link
            of that kind carried during this call. All links run in parallel, so the slowest link
            sets the time — see `timing.py`.
    """

    op: str
    label: str
    numel: int
    itemsize: int
    max_link_bytes: dict[str, int] = field(default_factory=dict)


class CommLog:
    """Exact per-rank and per-link byte counters for every send any collective makes."""

    def __init__(self, world_size: int) -> None:
        """Start every counter at zero."""
        self.size = world_size
        self.reset()

    def reset(self) -> None:
        """Zero every counter, e.g. between steps or between experiments."""
        self.sent = [0] * self.size
        self.received = [0] * self.size
        self.sent_by_op: dict[str, list[int]] = defaultdict(lambda: [0] * self.size)
        self.link_bytes: dict[tuple[int, int], int] = defaultdict(int)
        self.calls: list[CallRecord] = []

    def send(self, op: str, src: int, dst: int, nbytes: int) -> None:
        """Count one point-to-point message."""
        self.sent[src] += nbytes
        self.received[dst] += nbytes
        self.sent_by_op[op][src] += nbytes
        self.link_bytes[(src, dst)] += nbytes


class Rank:
    """One simulated device: an index, a node, its tensors, and the ledger of them."""

    def __init__(self, index: int, node: int) -> None:
        """Create an empty device.

        Args:
            index: The rank, `0 .. world_size - 1`.
            node: Which machine it sits on.
        """
        self.index = index
        self.node = node
        self.ledger = MemoryLedger()
        self._tensors: dict[tuple[str, str], torch.Tensor] = {}
        # storage key -> [bytes, how many held tensors share it, category charged]; see `put`.
        self._storages: dict[int, list] = {}
        self._storage_of: dict[tuple[str, str], int] = {}

    def put(self, category: str, name: str, tensor: "torch.Tensor") -> "torch.Tensor":
        """Store `tensor` on this device and charge **its storage's** bytes to `category`.

        **Storage, not view.** A shard taken as a view of a full buffer looks like 1/N of the bytes
        and keeps all N/N alive. Charging `numel × element_size` would report the view; this charges
        `untyped_storage().nbytes()`, the memory actually held. It was the view for the first
        version, and removing a `.clone()` from a shard's placement then left the full fp32 buffer
        alive while the ledger reported 1/N of it — with every test green.

        **Counted once per device.** If a second held tensor shares a storage already on this
        device, it is charged nothing more: the bytes exist once. They are credited back — to the
        category that was charged for them — when the last tensor sharing them is dropped.

        Raises:
            KeyError: If something is already stored under that name — a silent overwrite would
                leave the old bytes charged with nothing holding them.
        """
        key = (category, name)
        if key in self._tensors:
            raise KeyError(f"rank {self.index} already holds {category}/{name}")
        storage, nbytes = _footprint(tensor)
        entry = self._storages.get(storage)
        if entry is None:
            self._storages[storage] = [nbytes, 1, category]
            charged = nbytes
        else:
            entry[1] += 1
            charged = 0
        self._tensors[key] = tensor
        self._storage_of[key] = storage
        self.ledger.allocate(category, name, charged)
        return tensor

    def get(self, category: str, name: str) -> "torch.Tensor":
        """The tensor stored under `category/name`."""
        return self._tensors[(category, name)]

    def drop(self, category: str, name: str) -> None:
        """Free `category/name`: remove this device's reference and credit its bytes back.

        After this the rank holds no reference to the tensor at all, so if nothing else does
        either, Python frees it. `tests/test_zerosim_stages.py` checks that with a weak reference
        — a ledger that said "freed" while the tensor stayed alive would be a ledger that lies.
        """
        key = (category, name)
        del self._tensors[key]
        storage = self._storage_of.pop(key)
        entry = self._storages[storage]
        entry[1] -= 1
        if entry[1] > 0:
            self.ledger.free(category, name, 0)  # the storage lives on in another held tensor
            return
        del self._storages[storage]
        self.ledger.free(entry[2], name, entry[0])

    def holds(self, category: str, name: str) -> bool:
        """Whether `category/name` is currently stored here."""
        return (category, name) in self._tensors

    def held(self) -> list[tuple[str, str, "torch.Tensor"]]:
        """Everything this device holds now, as `(category, name, tensor)`."""
        return [(cat, name, tensor) for (cat, name), tensor in self._tensors.items()]


def _footprint(tensor) -> tuple[int, int]:
    """`(storage identity, storage bytes)` — what holding `tensor` actually keeps alive.

    A torch tensor answers with its untyped storage's address and size. Anything else (the tests'
    torch-free stand-ins) answers with its own identity and `numel × element_size`.
    """
    storage = getattr(tensor, "untyped_storage", None)
    if storage is None:
        return id(tensor), tensor.numel() * tensor.element_size()
    held = storage()
    return held.data_ptr(), held.nbytes()


class World:
    """The simulated cluster: ranks, their nodes, and what kind of link joins each pair.

    Args:
        world_size: Number of ranks.
        devices_per_node: Ranks per node. Defaults to all of them on one node.
    """

    def __init__(self, world_size: int, devices_per_node: int | None = None) -> None:
        """Lay the ranks out in order across nodes."""
        per_node = devices_per_node or world_size
        if world_size < 1 or world_size % per_node:
            raise ValueError(f"{world_size} ranks do not fill nodes of {per_node}")
        self.size = world_size
        self.devices_per_node = per_node
        self.ranks = [Rank(r, r // per_node) for r in range(world_size)]
        self.comm = CommLog(world_size)

    def link_kind(self, src: int, dst: int) -> str:
        """`"intra"` if both ranks share a node, else `"inter"`."""
        return "intra" if self.ranks[src].node == self.ranks[dst].node else "inter"

    def ring_links(self) -> list[tuple[int, int, str]]:
        """Every link the ring uses, as `(src, dst, kind)` — rank `r` sends only to `r + 1`."""
        return [
            (r, (r + 1) % self.size, self.link_kind(r, (r + 1) % self.size))
            for r in range(self.size)
        ]

    def persistent_bytes(self, rank: int) -> int:
        """Bytes rank `rank` holds right now in the persistent categories."""
        current = self.ranks[rank].ledger.current
        return sum(current[c] for c in PERSISTENT)
