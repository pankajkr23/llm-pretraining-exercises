"""Explain a variant from its own code: what changed, with what numbers, and how far to trust them.

Everything here is derived at call time — the diff from `inspect.getsource`, the shapes from a live
forward pass, the state size from the tensors the layer really keeps, the trust from the ledger —
so the documentation cannot describe code that no longer exists. `tools/build_lab_docs.py` renders
every variant into `docs/ATTENTION_LAB.md`; the notebook calls `describe(name)` before running one.
"""

import difflib
import inspect
from typing import Any

import torch

from attention.lab import hparams, registry
from attention.lab.base import MixerSpec, as_kwargs

TRUST_LABEL = {
    "verified": "verified",
    "ours": "our choice",
    "not verified": "NOT VERIFIED",
    "failed": "FAILED VERIFICATION",
}


def _class_of(spec: MixerSpec) -> type:
    torch.manual_seed(0)
    return type(spec.build("lab"))


def _forward_source(cls: type) -> str:
    """The source of every method the class defines itself, in definition order."""
    parts = []
    for name, member in cls.__dict__.items():
        if inspect.isfunction(member) and (name == "__init__" or not name.startswith("__")):
            parts.append(inspect.getsource(member))
    return "\n".join(parts) if parts else inspect.getsource(cls)


def code_diff(spec: MixerSpec, context: int = 2) -> str:
    """A unified diff of this variant's class against its parent's, or its whole source for a root.

    Variants that share one class and differ only in parameters get a one-line note instead of an
    empty diff, because "same code, different numbers" is itself the lesson (GQA vs MHA).
    """
    mine = _class_of(spec)
    if spec.parent is None:
        return _forward_source(mine)
    theirs = _class_of(registry.get(spec.parent))
    if mine is theirs:
        return (
            f"# same class as `{spec.parent}` ({mine.__name__}); only the parameters below differ"
        )
    lines = difflib.unified_diff(
        _forward_source(theirs).splitlines(),
        _forward_source(mine).splitlines(),
        fromfile=f"{spec.parent} ({theirs.__name__})",
        tofile=f"{spec.name} ({mine.__name__})",
        n=context,
        lineterm="",
    )
    return "\n".join(lines)


def config_rows(spec: MixerSpec) -> list[dict[str, Any]]:
    """One row per parameter: lab value, paper value, source, trust, and why."""
    lab = {p.name: p for p in spec.lab}
    paper = {p.name: p for p in spec.paper}
    rows = []
    for name in list(lab) + [n for n in paper if n not in lab]:
        shown = paper.get(name) or lab[name]
        rows.append(
            {
                "name": name,
                "meaning": shown.meaning,
                "lab": lab[name].value if name in lab else "",
                "paper": paper[name].value if name in paper else "",
                "source": shown.source,
                "trust": _trust_label(shown),
                "note": shown.note or _quote_for(shown.source),
            }
        )
    return rows


def _trust_label(param) -> str:
    """The trust class, marked when the relevance judgement asked for a human spot-check."""
    label = TRUST_LABEL[hparams.trust(param)]
    record = hparams.ledger().get(param.source, {})
    if label == "verified" and record.get("relevance_note", "").startswith("SPOT-CHECK"):
        return "verified — flagged for review"
    return label


def _quote_for(source: str) -> str:
    """The quote, where it is, and the relevance note that says how to read it."""
    record = hparams.ledger().get(source)
    if not record:
        return ""
    text = f"“{record['quote']}” ({record['where']})"
    note = record.get("relevance_note", "").removeprefix("SPOT-CHECK: ")
    return f"{text} — {note}" if note and not note.startswith("The sentence states") else text


def shapes(spec: MixerSpec, tokens: int = 8, batch: int = 2) -> dict[str, list[int]]:
    """Input, output and every state tensor's shape after a live forward pass at lab scale."""
    torch.manual_seed(0)
    mixer = spec.build("lab").eval()
    width = as_kwargs(spec.lab)["d_model"]
    x = torch.randn(batch, tokens, width)
    memory = None
    if spec.cross:
        memory = torch.randn(batch, 5, as_kwargs(spec.lab).get("d_memory", width))
    with torch.no_grad():
        y, state = mixer(x, memory=memory)
    out = {"input": list(x.shape), "output": list(y.shape)}
    if isinstance(state, dict):
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                out[f"state.{key}"] = list(value.shape)
    return out


def state_curve(spec: MixerSpec, lengths: tuple[int, ...] = (16, 64, 256)) -> list[tuple[int, int]]:
    """Bytes the layer keeps after each prefill length, counted from its real state."""
    torch.manual_seed(0)
    mixer = spec.build("lab").eval()
    width = as_kwargs(spec.lab)["d_model"]
    rows = []
    for n in lengths:
        memory = None
        if spec.cross:
            memory = torch.randn(1, 5, as_kwargs(spec.lab).get("d_memory", width))
        with torch.no_grad():
            _, state = mixer(torch.randn(1, n, width), memory=memory)
        rows.append((n, mixer.state_bytes(state)))
    return rows


def describe(name: str) -> str:
    """The full Markdown section for one variant."""
    spec = registry.get(name)
    lines = [f"### `{spec.name}`", ""]
    lines.append(
        f"**Family:** {spec.family} · **starts from:** `{spec.parent or '—'}` · "
        f"**covers:** `{spec.covers or 'lab-only'}` · **state:** {spec.state_growth}"
    )
    lines += ["", spec.summary, "", f"**Checked against:** {spec.checked_against}", ""]
    lines += [
        "**The change**",
        "",
        "```diff" if spec.parent else "```python",
        code_diff(spec),
        "```",
    ]
    lines += [
        "",
        "**Configuration**",
        "",
        "| parameter | meaning | lab | paper | trust | source |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in config_rows(spec):
        source = row["note"] or row["source"]
        lines.append(
            f"| `{row['name']}` | {row['meaning']} | {row['lab']} | {row['paper']} | "
            f"{row['trust']} | {source.replace('|', '/')} |"
        )
    lines += ["", "**Shapes at lab scale** (batch 2, 8 tokens)", ""]
    lines += [f"- `{k}`: {v}" for k, v in shapes(spec).items()]
    lines += ["", "**State kept, by prefill length**", ""]
    lines += [f"- {n} tokens: {b:,} bytes" for n, b in state_curve(spec)]
    if spec.exempt:
        lines += ["", "**Generic checks that do not apply, and why**", ""]
        lines += [f"- `{check}`: {why}" for check, why in spec.exempt.items()]
    return "\n".join(lines) + "\n"
