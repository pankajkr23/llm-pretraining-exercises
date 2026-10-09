"""Measuring memory two ways, and a loss that does not let the output head dominate it.

**Bytes saved for backward, counted by storage.** `saved_bytes` runs a forward pass inside
PyTorch's `saved_tensors_hooks` and sums the bytes of the distinct storages the hooks are shown.
That is the activation memory a step holds at the end of its forward pass — the quantity
reversibility exists to shrink — measured the same way on CPU, CUDA and Apple's MPS, which has no
peak-memory counter of its own. What the hooks see, and what they do not:

- **Seen:** every tensor a built-in operation saves; everything a custom `autograd.Function` passes
  to `ctx.save_for_backward`; and the inputs of a non-reentrant `torch.utils.checkpoint` region,
  which torch 2.13 keeps as saved tensors (so the chunked loss's input, the final hidden states, is
  counted — a test fails if a later torch stops doing this).
- **Not seen:** a tensor stored on `ctx` as a plain attribute, which is why a test requires the
  reversible `Function` to keep nothing on `ctx` but modules and numbers; and anything allocated
  during the backward pass. For the reversible stack the second matters: the backward pass re-runs
  one block with autograd on, and that block's own saved tensors are live while it does.
  `experiments.max_batch` measures that working set separately and adds it to the reversible
  model's per-sequence cost. Gradient buffers, the allocator's slack and kernel workspaces are not
  counted for either variant.

**The largest batch that actually trains, found by trying.** `fits` runs one full training step
(forward, backward, optimiser) under a hard GPU memory cap and reports whether it raised an
out-of-memory error; `largest` doubles the batch until a step fails, then bisects. On a device that
cannot be capped (a CPU), the search falls back to the saved-bytes measurement: the batch at which
model, gradients, optimiser state and the measured activations reach the budget.

**Every variant uses the same chunked, recomputed loss.** At large batch the logits — `tokens ×
10,001` floats — outgrow the residual stack's activations, and reversibility saves only the stack.
`chunked_loss` projects and scores `chunk` rows at a time under `torch.utils.checkpoint`, so only
one chunk's logits exist during the forward pass and each is recomputed during the backward. For
scale: at 32,768 rows (128 sequences of 256 tokens), 32,768 × 10,001 float32 logits are
1,250 MiB, which a plain cross-entropy keeps for its backward pass. A test checks, by watching the
logits' storages die rather than through the hooks, that the chunked loss keeps none of them.
Without this, the head would set the maximum batch for every variant and the comparison would
measure the loss, not the stack.
"""

from collections.abc import Callable

import torch
from torch.nn import functional
from torch.utils.checkpoint import checkpoint

GIB = 2**30


def _chunk_loss(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return functional.cross_entropy(hidden @ weight.T, targets, reduction="sum")


def chunked_loss(
    hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, chunk: int
) -> torch.Tensor:
    """Mean cross-entropy of `hidden @ weight.T` on `targets`, `chunk` rows at a time, recomputed.

    Args:
        hidden: `(n, d)` final hidden states.
        weight: `(vocab, d)` output head weight.
        targets: `(n,)` target ids.
        chunk: Rows per chunk.

    Returns:
        The same scalar as `cross_entropy(hidden @ weight.T, targets)`.
    """
    if chunk <= 0:
        raise ValueError(f"chunk must be positive, got {chunk}")
    total = hidden.new_zeros(())
    for start in range(0, hidden.shape[0], chunk):
        piece = slice(start, start + chunk)
        if torch.is_grad_enabled():
            total = total + checkpoint(
                _chunk_loss, hidden[piece], weight, targets[piece], use_reentrant=False
            )
        else:
            total = total + _chunk_loss(hidden[piece], weight, targets[piece])
    return total / hidden.shape[0]


def saved_bytes(forward: Callable[[], torch.Tensor]) -> int:
    """Bytes of the distinct tensors kept for backward by `forward()`, counted by storage.

    Two saved views of one storage are counted once; a tensor whose storage is an input the caller
    already holds (a parameter, a batch) is counted too. Compare variants by the difference between
    two batch sizes, in which those fixed inputs cancel. The module docstring lists what the hooks
    cannot see: tensors kept as `ctx` attributes, and anything allocated during backward.

    Args:
        forward: Runs a forward pass and returns the loss. The loss is discarded.

    Returns:
        Total bytes.
    """
    storages: dict[int, int] = {}
    held: list[torch.Tensor] = []

    # The graph keeps nothing; this list keeps the saved tensors alive until the forward ends.
    # Returning the tensor to autograd leaked the whole forward graph: the reversible Function saves
    # its own outputs, an output held in the graph keeps its autograd node alive, and that node's
    # ctx holds the saved states and the blocks — a cycle through C++ objects Python's collector
    # cannot break. Every run leaked one graph, and three rate checks in one memory-capped process
    # ran it out of memory. Holding nothing at all fails differently: each saved tensor is freed at
    # once, its memory is reused, and a later tensor at the same address overwrites its count. A
    # plain list held outside the graph does neither, and is cleared before returning. The forward
    # is never backwarded, so unpacking is refused rather than answered with a wrong tensor.
    def pack(tensor: torch.Tensor) -> None:
        storage = tensor.untyped_storage()
        storages[storage.data_ptr()] = storage.nbytes()
        held.append(tensor)

    def unpack(_: None) -> torch.Tensor:
        raise RuntimeError("saved_bytes measures a forward pass; it cannot be backwarded")

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        loss = forward()
    del loss
    held.clear()
    return sum(storages.values())


def state_bytes(model: torch.nn.Module) -> int:
    """Parameters, their gradients, and AdamW's two moments, all fp32: 16 bytes per parameter."""
    return 4 * sum(p.numel() * p.element_size() for p in model.parameters())


def device_limit_bytes(device: str) -> int | None:
    """What the backend says this process could use on the GPU at most; None on a CPU.

    Recorded beside the budget so a reader can see how much of the machine the cap leaves unused.
    """
    if device == "mps":
        return int(torch.mps.recommended_max_memory())
    if device == "cuda":
        return int(torch.cuda.get_device_properties(0).total_memory)
    return None


def is_oom(error: BaseException) -> bool:
    """Whether an exception is an allocator running out of memory, on any backend."""
    if isinstance(error, torch.OutOfMemoryError):
        return True
    return "out of memory" in str(error).lower()


def cap(device: str, budget_gib: float) -> bool:
    """Limit this process's GPU memory to `budget_gib`; False on a device that cannot be capped.

    The cap lasts for the life of the process, so experiments call this only in a process that does
    nothing afterwards but run at the batch the search found.
    """
    if device == "mps":
        torch.mps.set_per_process_memory_fraction(
            budget_gib * GIB / torch.mps.recommended_max_memory()
        )
        return True
    if device == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, budget_gib * GIB / total))
        return True
    return False


def release(device: str) -> None:
    """Return cached blocks to the device after a failed or finished trial."""
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def fits(step: Callable[[int], None], batch: int, device: str) -> bool:
    """Whether one training step at `batch` completes without running out of memory."""
    try:
        step(batch)
        return True
    except (RuntimeError, torch.OutOfMemoryError) as error:
        if not is_oom(error):
            raise
        return False
    finally:
        release(device)


def largest(predicate: Callable[[int], bool], start: int, ceiling: int) -> int:
    """The largest batch for which `predicate` holds: double from `start`, then bisect.

    Args:
        predicate: `batch -> fits?`, assumed monotone (if a batch fits, every smaller one does).
        start: A batch to begin from. If it does not fit, the search goes down instead.
        ceiling: Never test beyond this.

    Returns:
        The largest fitting batch, or 0 if not even a batch of 1 fits.
    """
    if not predicate(1):
        return 0
    low = 1
    high = None
    probe = max(1, min(start, ceiling))
    while probe <= ceiling:
        if predicate(probe):
            low = probe
            if probe == ceiling:
                return ceiling
            probe = min(probe * 2, ceiling)
        else:
            high = probe
            break
    if high is None:
        return low
    while high - low > 1:
        middle = (low + high) // 2
        if predicate(middle):
            low = middle
        else:
            high = middle
    return low


def analytic_largest(fixed_bytes: int, bytes_per_sample: float, budget_gib: float) -> int:
    """The batch at which fixed state plus `batch × bytes_per_sample` reaches the budget."""
    room = budget_gib * GIB - fixed_bytes
    return max(0, int(room // bytes_per_sample)) if bytes_per_sample > 0 else 0
