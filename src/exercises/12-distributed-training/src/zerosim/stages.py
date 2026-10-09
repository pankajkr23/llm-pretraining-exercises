"""Data parallelism and ZeRO stages 1, 2 and 3, as one training step each, on the simulated world.

**One engine, four policies.** Every stage runs the same forward, the same backward and the same
AdamW arithmetic on the same flat-buffer partitioning (`flat.py`). What a stage changes is only
*what each rank keeps* and *which collective moves the gradients and the weights*:

- **Stage 0, data parallelism.** Every rank keeps everything. Gradients are all-reduced, so every
  rank gets the full average, and every rank updates every weight itself.
- **Stage 1, shard the optimiser state.** Every rank keeps the weights and gradients, but only
  **1/N of the master copy, m and v**. Gradients are reduce-scattered, because rank `r` needs only
  its own slice; it updates that slice, and an all-gather spreads the new weights.
- **Stage 2, also shard the gradients.** As stage 1, but the reduce-scatter happens **unit by unit
  during the backward pass**, and each unit's full gradient is freed as soon as it has been
  scattered. A rank keeps 1/N of the gradients.
- **Stage 3, also shard the weights.** A rank keeps **1/N of everything**. It updates its slice and
  keeps only that.

Stage 3 has no full copy of any unit's weights between uses. It all-gathers a unit's weights just
before that unit's forward, frees them, and all-gathers them again just before that unit's backward.

**The step runs in lockstep, unit by unit, across all ranks.** In a real cluster every device runs
its own process and they meet at each collective. Here one process plays all of them, so it runs
unit 0 on every rank, then unit 1 on every rank, and so on — which puts every rank at the same
point when a collective needs them all, exactly as the real barrier does.

**Each unit's forward is recomputed inside the backward**, on every stage alike. Only each unit's
*input* is kept from the forward pass. The reason is honesty about memory: PyTorch's autograd graph
keeps a reference to every weight the forward used, so a ZeRO-3 rank that "freed" its gathered
weights while keeping the graph would hold them anyway, and the ledger would be lying. Recomputing
from the unit's input lets the gathered weights genuinely die — a weak-reference test proves they
do. It is unit-level activation checkpointing, applied identically to every stage so it cannot
distort the comparison; it doubles the forward compute of every stage and changes no result.
"""

import torch
from torch.utils.flop_counter import FlopCounterMode

from . import flat
from .adamw import Hyper, adamw_step
from .collectives import all_gather, all_reduce, reduce_scatter
from .config import Config
from .model import DemoModel, build_model, split
from .precision import recipe
from .world import World

STAGES = (0, 1, 2, 3)
"""Stage 0 is plain data parallelism; 1–3 are the ZeRO stages."""

STAGE_NAMES = {0: "DP (stage 0)", 1: "ZeRO-1", 2: "ZeRO-2", 3: "ZeRO-3"}


class Trainer:
    """One stage, one simulated world, one model, trained step by step.

    Args:
        config: World size, nodes, model, optimiser, precision mode.
        stage: 0, 1, 2 or 3.
        model: A pre-built model, so several trainers can share the same initial weights. Built
            from `config` when omitted.
        count_flops: Count each rank's forward and backward FLOPs with torch's own counter. Slower,
            so off unless a measurement needs it.
    """

    def __init__(
        self,
        config: Config,
        stage: int,
        model: DemoModel | None = None,
        count_flops: bool = False,
    ) -> None:
        """Lay the model out on every rank as this stage requires."""
        if stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}, got {stage}")
        self.config = config
        self.stage = stage
        self.recipe = recipe(config.mode)
        self.param_dtype = self.recipe.torch_dtype("param")
        self.grad_dtype = self.recipe.torch_dtype("grad")
        self.world = World(config.world_size, config.devices_per_node)
        self.model = model or build_model(config)
        self.layouts = [
            flat.layout(unit.name, unit.shapes, config.world_size) for unit in self.model.units
        ]
        self.hyper = Hyper(
            config.learning_rate, config.beta1, config.beta2, config.eps, config.weight_decay
        )
        self.steps_taken = 0
        self.count_flops = count_flops
        n = config.world_size
        self.flops = {
            "forward": [0] * n,
            "recompute": [0] * n,
            "backward": [0] * n,
        }
        self.optimizer_elements = [0] * n
        self.losses: list[list[float]] = []
        #: Bytes each rank sent during each step, by collective and in total — recorded per step so
        #: "every step sends the same" is a comparison, not an inference from a divisible total.
        self.comm_by_step: list[dict[str, list[int]]] = []
        self._traffic_so_far: dict[str, list[int]] = {}
        self._place_state()

    # --- layout ---------------------------------------------------------------------------------

    @property
    def n(self) -> int:
        """World size."""
        return self.world.size

    def _place_state(self) -> None:
        """Put each category on each rank, full or as a 1/N shard, as this stage dictates."""
        for unit, spec in zip(self.model.units, self.layouts, strict=True):
            full = flat.flatten(unit.initial, spec, torch.float32)
            for rank in self.world.ranks:
                mine = flat.shard_of(full, rank.index, self.n)
                # The compute copy of the weights: full until stage 3.
                source = mine if self.stage == 3 else full
                rank.put("params", unit.name, source.to(self.param_dtype).clone())
                # Gradients: full until stage 2.
                size = spec.shard if self.stage >= 2 else spec.padded
                rank.put("grads", unit.name, torch.zeros(size, dtype=self.grad_dtype))
                # Optimiser state: full only for stage 0.
                state = full if self.stage == 0 else mine
                if self.recipe.has_master:
                    rank.put("master", unit.name, state.clone())
                rank.put("adam_m", unit.name, torch.zeros_like(state))
                rank.put("adam_v", unit.name, torch.zeros_like(state))

    # --- one step -------------------------------------------------------------------------------

    def step(self, tokens: torch.Tensor) -> list[float]:
        """One optimiser step. `tokens[r]` is rank r's micro-batch, `[micro_batch, seq_len]`.

        Returns:
            Each rank's loss on its own micro-batch, before the update.
        """
        if tokens.shape[0] != self.n:
            raise ValueError(f"need one micro-batch per rank ({self.n}), got {tokens.shape[0]}")
        inputs, targets = zip(*(split(tokens[r]) for r in range(self.n)), strict=True)
        last = len(self.model.units) - 1

        # Forward: keep only each unit's input.
        saved: list[list[torch.Tensor]] = []
        hidden = list(inputs)
        for index in range(len(self.model.units)):
            saved.append(hidden)
            weights = self._weights(index, "forward")
            outputs = []
            for r in range(self.n):
                outputs.append(self._forward(index, weights[r], hidden[r], targets[r], r))
            del weights
            self._release(index)
            hidden = outputs
        losses = [float(loss) for loss in hidden]
        self.losses.append(losses)

        # Backward, last unit first.
        upstream: list[torch.Tensor | None] = [None] * self.n
        for index in range(last, -1, -1):
            weights = self._weights(index, "backward")
            grads = []
            for r in range(self.n):
                grad_w, upstream[r] = self._backward(
                    index, weights[r], saved[index][r], targets[r], upstream[r], r
                )
                grads.append(grad_w)
            del weights
            self._release(index)
            self._take_gradients(index, grads)
            del grads

        self._finish_gradients()
        self._update()
        self._record_step_traffic()
        return losses

    def _record_step_traffic(self) -> None:
        """Append what each rank sent during this step: the counters' growth since the last one."""
        comm = self.world.comm
        now = {"total": list(comm.sent), **{op: list(v) for op, v in comm.sent_by_op.items()}}
        before = self._traffic_so_far
        self.comm_by_step.append(
            {
                key: [a - b for a, b in zip(values, before.get(key, [0] * self.n), strict=True)]
                for key, values in now.items()
            }
        )
        self._traffic_so_far = now

    # --- forward and backward on one rank --------------------------------------------------------

    def _forward(self, index: int, weights: torch.Tensor, inputs, targets, rank: int):
        spec = self.layouts[index]
        with torch.no_grad(), self._counter(rank, "forward"):
            return self.model.run_unit(index, flat.views(weights, spec), inputs, targets)

    def _backward(self, index: int, weights: torch.Tensor, inputs, targets, upstream, rank: int):
        """Recompute unit `index` from its saved input, then backpropagate through it.

        Returns:
            `(gradient of the unit's flat weights, gradient of the unit's input or None)`.
        """
        spec = self.layouts[index]
        leaf = weights.detach().requires_grad_(True)
        if index > 0:
            inputs = inputs.detach().requires_grad_(True)
        with self._counter(rank, "recompute"):
            out = self.model.run_unit(index, flat.views(leaf, spec), inputs, targets)
        with self._counter(rank, "backward"):
            if upstream is None:
                # The loss. Scaling by 1/N here makes the summed gradient the global mean.
                (out * (1.0 / self.n)).backward()
            else:
                out.backward(upstream)
        return leaf.grad, (inputs.grad if index > 0 else None)

    def _counter(self, rank: int, phase: str):
        if not self.count_flops:
            return _Null()
        return _Count(self.flops[phase], rank)

    # --- where the weights come from -------------------------------------------------------------

    def _weights(self, index: int, phase: str) -> list[torch.Tensor]:
        """Each rank's full weights for unit `index`: its own copy, or — stage 3 — gathered now."""
        name = self.model.units[index].name
        if self.stage < 3:
            return [rank.get("params", name) for rank in self.world.ranks]
        spec = self.layouts[index]
        outputs = [
            rank.put("gathered_params", name, torch.empty(spec.padded, dtype=self.param_dtype))
            for rank in self.world.ranks
        ]
        shards = [rank.get("params", name) for rank in self.world.ranks]
        return all_gather(self.world, shards, outputs, label=f"params/{name}/{phase}")

    def _release(self, index: int) -> None:
        """Stage 3 frees the gathered weights the moment the unit is done with them."""
        if self.stage == 3:
            name = self.model.units[index].name
            for rank in self.world.ranks:
                rank.drop("gathered_params", name)

    # --- gradients -------------------------------------------------------------------------------

    def _take_gradients(self, index: int, grads: list[torch.Tensor]) -> None:
        """Stages 0–1 write into a full gradient buffer; 2–3 reduce-scatter this unit right now."""
        name = self.model.units[index].name
        if self.stage < 2:
            for rank, grad in zip(self.world.ranks, grads, strict=True):
                rank.get("grads", name).copy_(grad)
            return
        buckets = [
            rank.put("grad_bucket", name, grad)
            for rank, grad in zip(self.world.ranks, grads, strict=True)
        ]
        shards = reduce_scatter(self.world, buckets, label=f"grads/{name}")
        del buckets
        for rank, shard in zip(self.world.ranks, shards, strict=True):
            rank.get("grads", name).copy_(shard)
            rank.drop("grad_bucket", name)

    def _finish_gradients(self) -> None:
        """Stage 0 all-reduces every gradient; stage 1 reduce-scatters it. 2–3 are done already."""
        if self.stage >= 2:
            return
        for unit in self.model.units:
            buffers = [rank.get("grads", unit.name) for rank in self.world.ranks]
            if self.stage == 0:
                all_reduce(self.world, buffers, label=f"grads/{unit.name}")
                continue
            shards = reduce_scatter(self.world, buffers, label=f"grads/{unit.name}")
            for rank, shard in zip(self.world.ranks, shards, strict=True):
                flat.shard_of(rank.get("grads", unit.name), rank.index, self.n).copy_(shard)

    # --- the update ------------------------------------------------------------------------------

    def _update(self) -> None:
        self.steps_taken += 1
        for unit in self.model.units:
            for rank in self.world.ranks:
                weight, grad = self._optimizer_view(rank, unit.name)
                adamw_step(
                    weight,
                    grad,
                    rank.get("adam_m", unit.name),
                    rank.get("adam_v", unit.name),
                    self.steps_taken,
                    self.hyper,
                )
                self.optimizer_elements[rank.index] += weight.numel()
            self._refresh_weights(unit.name)

    def _optimizer_view(self, rank, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """The fp32 weights this rank updates, and the averaged gradient for exactly those."""
        grads = rank.get("grads", name)
        if self.stage == 1:
            grads = flat.shard_of(grads, rank.index, self.n)
        if self.recipe.has_master:
            return rank.get("master", name), grads
        params = rank.get("params", name)
        if self.stage in (1, 2):
            params = flat.shard_of(params, rank.index, self.n)
        return params, grads

    def _refresh_weights(self, name: str) -> None:
        """Bring the compute copy of the weights up to date after the update."""
        ranks = self.world.ranks
        if self.stage in (1, 2):
            if self.recipe.has_master:
                for rank in ranks:
                    own = flat.shard_of(rank.get("params", name), rank.index, self.n)
                    own.copy_(rank.get("master", name))
            buffers = [rank.get("params", name) for rank in ranks]
            shards = [flat.shard_of(b, r, self.n) for r, b in enumerate(buffers)]
            all_gather(self.world, shards, buffers, label=f"params/{name}/update")
        elif self.recipe.has_master:  # stage 0 (full) and stage 3 (shard): a local cast
            for rank in ranks:
                rank.get("params", name).copy_(rank.get("master", name))

    # --- inspection (reads the simulator's state; not a collective, not counted) ---------------

    def full_weights(self, which: str = "optimizer") -> dict[str, torch.Tensor]:
        """Each unit's whole weight vector, unpadded, read straight out of the ranks.

        Args:
            which: `"optimizer"` for the fp32 weights AdamW updates (master copy, or the fp32
                weights themselves), `"params"` for the compute copy.

        This **reads** the simulated devices' memory from outside — it is how a test checks the
        result, not something a real cluster could do for free, and it moves no counted bytes.
        """
        if which not in ("optimizer", "params"):
            raise ValueError(f"which must be 'optimizer' or 'params', got {which!r}")
        category = "master" if which == "optimizer" and self.recipe.has_master else "params"
        # The master copy is sharded from stage 1; the compute copy only at stage 3.
        sharded = self.stage >= 1 if category == "master" else self.stage == 3
        out = {}
        for unit, spec in zip(self.model.units, self.layouts, strict=True):
            if sharded:
                pieces = [rank.get(category, unit.name) for rank in self.world.ranks]
                whole = torch.cat(pieces)
            else:
                whole = self.world.ranks[0].get(category, unit.name)
            out[unit.name] = whole[: spec.numel].clone()
        return out


class _Null:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


class _Count:
    """Add the FLOPs counted inside the block to `totals[rank]`."""

    def __init__(self, totals: list[int], rank: int) -> None:
        self.totals = totals
        self.rank = rank
        self.mode = FlopCounterMode(display=False)

    def __enter__(self) -> None:
        self.mode.__enter__()

    def __exit__(self, *exc: object) -> None:
        self.mode.__exit__(*exc)
        self.totals[self.rank] += self.mode.get_total_flops()


def train(config: Config, stage: int, model: DemoModel | None = None, **kwargs) -> Trainer:
    """Build a trainer and run `config.steps` steps on the corpus. Returns the trainer."""
    from .model import batches

    trainer = Trainer(config, stage, model=model, **kwargs)
    data = batches(config)
    for s in range(config.steps):
        trainer.step(data[s])
    return trainer
