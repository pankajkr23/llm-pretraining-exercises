"""Every knob this exercise measures against, in one dataclass.

Three kinds of number live here, and they are kept apart on purpose because they deserve different
amounts of trust:

- **The simulated world and the demo model** — world size, topology, model shape, optimiser. These
  are choices. Change one and the measured results move; `provenance.config_fingerprint` moves
  with it.
- **The assumed hardware figures** — link bandwidths, device throughput, card capacity. These are
  **not measurements**. Nothing in this exercise touches a real interconnect or a real accelerator.
  They exist only so the *time model* and the *memory ladder* can be quoted in seconds and in
  "fits / does not fit" rather than in bytes alone, and every place that uses them says so.
- **The ladder's model size** — the large model the bytes-per-weight formulas are priced at, which
  is a hypothetical, not a run.
"""

from dataclasses import dataclass, field

from lossheads.config import Config as ModelConfig

MODES = ("bf16-mixed", "fp32")
"""The two precision recipes a run can use. See `precision.py` for what each one stores."""


def _demo_model() -> ModelConfig:
    """Exercise 09's model, shrunk until 32 ranks run a step in well under a second on a CPU.

    `vocab_size` is **not** shrunk and cannot be: exercise 09's tokenizer emits ids up to 10,000,
    and its corpus loader refuses a smaller vocabulary rather than letting the embedding lookup
    fail. So the two vocabulary-sized tables (token embedding and output head) dominate the
    parameter count — which is realistic for a small model and is reported, not hidden.

    `batch_size` is the **micro-batch per rank**. The global batch is `world_size` times it.
    """
    return ModelConfig(d_model=32, n_layer=2, n_head=2, seq_len=32, batch_size=1)


@dataclass(frozen=True)
class Config:
    """The simulated cluster, the demo model, the optimiser and the assumed hardware figures.

    Attributes:
        world_size: Simulated devices. Each one is a `Rank` object inside this one process.
        nodes: How the devices are grouped into machines. Ranks `0..7` are node 0 and so on, and
            the ring visits them in rank order, so exactly `nodes` ring links cross a node boundary.
        model: Exercise 09's model configuration. `batch_size` is the per-rank micro-batch.
        steps: Optimiser steps in the measured run.
        learning_rate: AdamW's step size.
        beta1: AdamW's first-moment decay.
        beta2: AdamW's second-moment decay.
        eps: AdamW's denominator floor.
        weight_decay: Decoupled weight decay, applied to every parameter.
        seed: Seeds the weights and the order the corpus is cut in.
        mode: `"bf16-mixed"` (bf16 weights and gradients, fp32 master copy and moments) or
            `"fp32"` (everything fp32, no separate master copy).
        intra_node_bandwidth: **Assumed**, bytes per second, one direction, for a link between two
            devices on the same node. An NVLink-class figure, chosen as a round number.
        inter_node_bandwidth: **Assumed**, bytes per second, one direction, for a link that leaves
            the node. An InfiniBand-class figure, chosen as a round number.
        device_flops: **Assumed** sustained throughput of one device, FLOP/s, used only to turn
            counted FLOPs into a compute time in the time model.
        card_bytes: **Assumed** memory capacity of one device for the ladder, in bytes. 80 × 10⁹,
            the decimal figure a card is sold as.
        ladder_params: The hypothetical large model the ladder prices, in weights.
        ladder_world_sizes: The device counts the ladder is evaluated at.
        source: Where these values come from.
    """

    world_size: int = 32
    nodes: int = 4
    model: ModelConfig = field(default_factory=_demo_model)
    steps: int = 4
    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    seed: int = 12
    mode: str = "bf16-mixed"
    intra_node_bandwidth: float = 450e9
    inter_node_bandwidth: float = 50e9
    device_flops: float = 400e12
    card_bytes: int = 80 * 10**9
    ladder_params: int = 30 * 10**9
    ladder_world_sizes: tuple[int, ...] = (8, 16, 32, 64)
    source: str = (
        "the world, model and optimiser are this exercise's own choices, sized so 32 simulated "
        "ranks run in seconds on a CPU; the model is exercise 09's trunk and head; the two link "
        "bandwidths, the device throughput and the card capacity are ASSUMED round figures for "
        "the time model and the memory ladder, never measurements"
    )

    def __post_init__(self) -> None:
        """Refuse a world that cannot be laid out, rather than simulating a different one."""
        if self.world_size < 1:
            raise ValueError(f"world_size must be at least 1, got {self.world_size}")
        if self.nodes < 1 or self.world_size % self.nodes:
            raise ValueError(
                f"{self.world_size} devices cannot be split evenly across {self.nodes} nodes"
            )
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")

    @property
    def devices_per_node(self) -> int:
        """Devices on one node: 32 across 4 nodes is 8 per node."""
        return self.world_size // self.nodes

    @property
    def global_batch(self) -> int:
        """Sequences consumed per optimiser step, across every rank."""
        return self.world_size * self.model.batch_size
