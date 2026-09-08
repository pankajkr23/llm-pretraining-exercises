"""Where a run writes what it read, what it built, what happened and what came out.

**One directory per run, numbered in pipeline order, so `ls` tells you the story.** The failure
this exists to prevent is the one this exercise has already paid for: a published comparison whose
driver lived in a scratch directory, whose optimisation was recorded nowhere, and whose numbers
could therefore not be aimed at, checked or defended. A bundle alone was not enough, because a
bundle records the *conclusion* of each stage and none of the material.

```
artifacts/runs/<date>-<config_fingerprint>/
  00-manifest.json           config, provenance, corpus facts, every digest
  01-input/                  what went in
    corpus.meta.json           per-lane tokens, [UNK] share, licence, digest, epochs
    tokens.seed<N>.npy         the exact id stream that seed consumed, in order
    digests.json               data_digest per seed + batching_version
  02-model/                  what was built
    <arm>.seed<N>.init.json    shapes, parameter count, weight digest at step zero
  03-train/                  what happened
    <arm>.seed<N>.trace.csv    per-step loss AND pre-clip gradient norm
  04-output/                 what came out
    <arm>.seed<N>.ckpt.pt      trained weights, for the arms worth keeping
    <arm>.seed<N>.ckpt.json    sidecar: digests, environment, final metrics
    arms.json                  the comparison table
  05-verify/                 whether it holds up
    audit.json                 an independent re-derivation (verify.py writes this)
```

**The directory is gitignored and that is deliberate.** `artifacts/` is regenerable output;
`results/` is the measured evidence a document renders, and what goes there is a decision a person
takes after reading a run rather than a side effect of running one. What crosses the line is the
*digest*, never the weights.

**`<run-id>` is `<date>-<config_fingerprint>`**, so two runs cannot collide and the settings are
legible in the name — a directory called `2026-09-08-a72bf6053187` can be matched against a bundle
without opening either.
"""

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-time only, never executed
    import torch

    from embeddings.experiment import Arm, RunConfig

STAGES = ("01-input", "02-model", "03-train", "04-output", "05-verify")
"""The numbered stages, in pipeline order. Created eagerly, so an empty one is visible as an empty
one rather than as an absence somebody has to interpret."""

KEEP_WEIGHTS = (
    "dense tied embedding",
    "v1 - Kronecker in, untied head",
    "tied + n-gram (one-hot positions)",
)
"""Which arms' trained weights are worth 23 MB each.

The control, the published bar, and the recommendation — the three a reader would want to load and
poke at. Keeping all ten at five seeds would be 1.2 GB of regenerable output to say the same thing.
"""


def run_id(config: "RunConfig", date: str) -> str:
    """`<date>-<config_fingerprint>`.

    Args:
        config: The settings, whose fingerprint goes in the name.
        date: `YYYY-MM-DD`. Passed in rather than read from a clock, so a caller that wants a
            deterministic id can have one and a test never depends on today.
    """
    return f"{date}-{config.fingerprint()}"


def weight_digest(*modules: "torch.nn.Module") -> str:
    """`sha256:` over every named parameter of every module, in name order.

    Names are hashed alongside the bytes, so renaming a parameter moves the digest — otherwise two
    models with the same weights under different names would be indistinguishable, and the digest
    would be vouching for less than it appears to.
    """
    digest = hashlib.sha256()
    for module in modules:
        for name, parameter in sorted(module.named_parameters()):
            digest.update(name.encode("utf-8"))
            digest.update(parameter.detach().cpu().numpy().tobytes())
    return "sha256:" + digest.hexdigest()


class RunDirectory:
    """The numbered directory for one run, written as the run goes rather than at the end.

    Written *as it goes* on purpose: a writer that ran at the end would lose everything if the run
    died at step 400 of 500, and this repository has already lost fifteen trained models to a
    driver that fell over on its final statement.
    """

    def __init__(self, root: Path, config: "RunConfig", date: str) -> None:
        """Create the directory and its stages.

        Args:
            root: Usually `<exercise>/artifacts/runs`.
            config: The settings for this run.
            date: `YYYY-MM-DD`, for the id.
        """
        self.config = config
        self.path = Path(root) / run_id(config, date)
        for stage in STAGES:
            (self.path / stage).mkdir(parents=True, exist_ok=True)

    def manifest(self, provenance: dict, corpus: dict, extra: dict | None = None) -> Path:
        """Write `00-manifest.json` — the file to open first."""
        payload = {
            "run_id": self.path.name,
            "config": asdict(self.config),
            "provenance": provenance,
            "corpus": corpus,
            **(extra or {}),
        }
        return _write_json(self.path / "00-manifest.json", payload)

    def input(self, seed: int, tokens: "torch.Tensor", digest: str) -> Path:
        """Write the exact id stream one seed consumed, plus its digest.

        The stream itself, not a summary of it: a digest says two runs differed and only the stream
        says how. It is `int32` on disk because the vocabulary is 10,000 wide and `int64` would
        double the file to say nothing.
        """
        import numpy as np

        path = self.path / "01-input" / f"tokens.seed{seed}.npy"
        np.save(path, tokens.numpy().astype(np.int32))
        digests_path = self.path / "01-input" / "digests.json"
        digests = json.loads(digests_path.read_text()) if digests_path.is_file() else {}
        digests[str(seed)] = {
            "data_digest": digest,
            "batching_version": self.config.batching_version,
            "shape": list(tokens.shape),
        }
        _write_json(digests_path, digests)
        return path

    def corpus_meta(self, corpus: dict) -> Path:
        """Write `01-input/corpus.meta.json` — the lane table, as a file of its own.

        Duplicated from the manifest deliberately: the manifest is the index and this is the input,
        and a reader asking "what text was this" should not have to know which key of a larger file
        to look under.
        """
        return _write_json(self.path / "01-input" / "corpus.meta.json", corpus)

    def initial(self, arm: "Arm", seed: int, trunk, head) -> Path:
        """Write what was built, before a single gradient step.

        The weight digest at step zero is what makes "two arms at one seed share their trunk" a
        checkable claim after the fact rather than only inside a test.
        """
        payload = {
            "arm": arm.name,
            "seed": seed,
            "v_free": arm.v_free,
            "parameters": sum(
                p.numel()
                for p in {id(p): p for p in [*trunk.parameters(), *head.parameters()]}.values()
            ),
            "shapes": {name: list(p.shape) for name, p in sorted(head.named_parameters())},
            "trunk_block_digest": weight_digest(trunk.blocks),
            "weight_digest": weight_digest(trunk, head),
        }
        return _write_json(
            self.path / "02-model" / f"{_slug(arm.name)}.seed{seed}.init.json", payload
        )

    def trace(self, result: dict) -> Path:
        """Write the per-step loss and pre-clip gradient norm as CSV.

        CSV rather than JSON because this is the one artefact a person opens in a spreadsheet, and
        it is the only place the *shape* of a run is visible — a bundle records where the loss
        ended and never how it got there.
        """
        path = self.path / "03-train" / f"{_slug(result['arm'])}.seed{result['seed']}.trace.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["step", "loss", "grad_norm"])
            for step, (loss, norm) in enumerate(
                zip(result["losses"], result["grad_norms"], strict=True)
            ):
                writer.writerow([step, f"{loss:.6f}", f"{norm:.6f}"])
        return path

    def checkpoint(self, result: dict, trunk, head, provenance: dict) -> Path | None:
        """Write trained weights and a sidecar, for the arms in `KEEP_WEIGHTS`.

        Returns:
            The checkpoint path, or `None` when this arm's weights are not kept — reported rather
            than silent, so a caller can say which arms were skipped and why.
        """
        if result["arm"] not in KEEP_WEIGHTS:
            return None
        import torch

        stem = f"{_slug(result['arm'])}.seed{result['seed']}"
        path = self.path / "04-output" / f"{stem}.ckpt.pt"
        torch.save({"trunk": trunk.state_dict(), "head": head.state_dict()}, path)
        _write_json(
            self.path / "04-output" / f"{stem}.ckpt.json",
            {
                "arm": result["arm"],
                "seed": result["seed"],
                "weights": path.name,
                "weight_digest": weight_digest(trunk, head),
                "file_digest": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
                "data_digest": result["data_digest"],
                "batching_version": result["batching_version"],
                "device": result["device"],
                "mean_last_50": result["mean_last_50"],
                "lane_loss": result["lane_loss"],
                "provenance": provenance,
            },
        )
        return path

    def arms(self, comparisons: list[dict]) -> Path:
        """Write the comparison table on its own, beside the weights it describes."""
        return _write_json(self.path / "04-output" / "arms.json", comparisons)


def _slug(name: str) -> str:
    """A filename-safe form of an arm name, stable across runs."""
    keep = [c if c.isalnum() else "-" for c in name.lower()]
    return "-".join(part for part in "".join(keep).split("-") if part)


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    return path
