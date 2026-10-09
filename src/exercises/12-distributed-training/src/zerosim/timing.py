"""A time model — **assumed figures applied to measured counts**. Not a measurement of anything.

Nothing in this exercise crosses a real wire or runs on a real accelerator, so no time here is
measured. What *is* measured is how many bytes each collective put on each ring link and how many
FLOPs each rank executed. This module turns those into seconds using three **assumed** constants
from `Config` — an intra-node link bandwidth, an inter-node link bandwidth and a device throughput
— so that a reader can see the *shape* of the trade-off in familiar units. Change the assumptions
and every time here changes; the byte and FLOP counts underneath do not.

**How a ring collective is timed.** All links send at once, one chunk per step, and a step ends
when the slowest link has delivered. So a collective takes `max over links (bytes on that link ÷
that link's bandwidth)`. With 32 devices on 4 nodes, the ring crosses a node boundary 4 times, and
those 4 slow links set the pace for all 32 — the fast intra-node links spend most of the collective
waiting. That is the real reason large clusters use hierarchical collectives, which this simulator
does not model.

**No overlap is assumed.** Real systems overlap communication with the backward pass; here compute
time and communication time are simply added, which is the pessimistic end.
"""

from .config import Config
from .world import CallRecord


def bandwidths(config: Config) -> dict[str, float]:
    """The two assumed link speeds, bytes per second, keyed by link kind."""
    return {"intra": config.intra_node_bandwidth, "inter": config.inter_node_bandwidth}


def collective_seconds(call: CallRecord, config: Config) -> float:
    """One collective's time: its busiest link of each kind, at that kind's assumed speed."""
    speed = bandwidths(config)
    return max((nbytes / speed[kind] for kind, nbytes in call.max_link_bytes.items()), default=0.0)


def comm_seconds(calls: list[CallRecord], config: Config) -> float:
    """Total communication time for a sequence of collectives run one after another."""
    return sum(collective_seconds(call, config) for call in calls)


def compute_seconds(flops: float, config: Config) -> float:
    """FLOPs at the assumed device throughput."""
    return flops / config.device_flops


def ring_seconds(payload_bytes: float, world_size: int, config: Config) -> float:
    """One reduce-scatter or all-gather of `payload_bytes`, predicted rather than counted.

    Each link carries `payload × (N−1)/N` bytes. If the ring spans more than one node, the slowest
    link is an inter-node one; otherwise every link is intra-node. Used for the large-model ladder,
    where nothing is simulated.
    """
    if world_size == 1:
        return 0.0
    per_link = payload_bytes * (world_size - 1) / world_size
    spans_nodes = world_size > config.devices_per_node
    speed = config.inter_node_bandwidth if spans_nodes else config.intra_node_bandwidth
    return per_link / speed
