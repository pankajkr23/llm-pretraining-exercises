"""Generate `RESULTS.md` from `results/zero.json`. No number in it is typed by anyone.

```bash
uv run python src/exercises/12-distributed-training/tools/render_results.py
```

Every figure, and every verdict word ("matches", "fits"), is a lookup into the bundle or a
comparison computed from it — never a literal in this template. `tests/test_zerosim_results.py`
re-renders and fails if the tracked `RESULTS.md` differs by a single byte, and it needs no `torch`,
so a stale document is caught in the ordinary CI job.

It also writes `web/data.js`, the page's only source of numbers: the bundle without its free-text
notes, plus `page_numbers()` — every ratio and verdict the page quotes, computed here once. The two
multiples both outputs print (`comm_multiple`, `optimizer_fewer`) are shared functions, so the
document and the page cannot disagree about them. The same test file checks `data.js` is fresh.
"""

import json
import sys
from pathlib import Path

EXERCISE = Path(__file__).resolve().parents[1]
RESULTS = EXERCISE / "results" / "zero.json"
OUT = EXERCISE / "RESULTS.md"

CATEGORIES = ("params", "grads", "master", "adam_m", "adam_v")
FORMULA_TEXT = {
    "bf16-mixed": "DP 16 · ZeRO-1 4 + 12/N · ZeRO-2 2 + 14/N · ZeRO-3 16/N",
    "fp32": "DP 16 · ZeRO-1 8 + 8/N · ZeRO-2 4 + 12/N · ZeRO-3 16/N",
}
"""The hand formulas, as text. `tests/test_zerosim_results.py` checks each against `formulas.py`."""

LABELS = {
    "params": "weights",
    "grads": "gradients",
    "master": "fp32 master",
    "adam_m": "Adam m",
    "adam_v": "Adam v",
}


def _n(value: float) -> str:
    """An integer-valued count with thousands separators."""
    return f"{int(value):,}"


def comm_multiple(stage_block: dict, world_size: int) -> float:
    """How many `P·(N−1)/N` one device sent per step: the counted total over one ring pass.

    Computed here once, and used by both `RESULTS.md` and the page's `data.js`, so the two cannot
    print different multiples of the same count.
    """
    comm = stage_block["communication"]
    return comm["per_step_sent"]["total"] / (comm["payload_bytes"] * (world_size - 1) / world_size)


def optimizer_fewer(stages: dict, key: str) -> float:
    """How many times fewer weights this stage's optimiser updates than data parallelism's."""
    return (
        stages["0"]["compute"]["optimizer_elements"] / stages[key]["compute"]["optimizer_elements"]
    )


def key_bias_drifted_further(eq: dict) -> bool:
    """Whether the 64 key-bias weights drifted further from the single device than all the rest."""
    return eq["fp32_weights_max_abs_key_bias"] > eq["fp32_weights_max_abs_except_key_bias"]


def _fewer(ratio: float) -> str:
    return "—" if ratio == 1 else f"{ratio:g}× fewer"


def _verdict(measured: float, predicted: float) -> str:
    return "matches" if measured == predicted else "**DIFFERS**"


def _memory_table(stages: dict) -> str:
    head = "| stage | " + " | ".join(LABELS[c] for c in CATEGORIES) + " | total | formula |"
    rule = "| --- |" + " ---: |" * (len(CATEGORIES) + 1) + " --- |"
    rows = []
    for key in sorted(stages):
        block = stages[key]["memory"]
        cells = " | ".join(_n(block["measured"][c]) for c in CATEGORIES)
        verdict = all(block["measured"][c] == block["predicted"][c] for c in CATEGORIES)
        rows.append(
            f"| {stages[key]['name']} | {cells} | **{_n(block['measured']['total'])}** | "
            f"{'every category matches' if verdict else '**DIFFERS**'} |"
        )
    return "\n".join([head, rule, *rows])


def _bpw_table(stages: dict) -> str:
    rows = [
        "| stage | measured | predicted | | peak incl. transients |",
        "| --- | ---: | ---: | --- | ---: |",
    ]
    for key in sorted(stages):
        bpw = stages[key]["memory"]["bytes_per_weight"]
        rows.append(
            f"| {stages[key]['name']} | {bpw['measured_padded']:g} | {bpw['predicted']:g} | "
            f"{_verdict(bpw['measured_padded'], bpw['predicted'])} | "
            f"{_n(stages[key]['memory']['peak_total'])} bytes |"
        )
    return "\n".join(rows)


def render(run: dict) -> str:
    """Build the whole document. Every interpolation is a lookup."""
    prov = run["provenance"]
    cfg = run["config"]
    model = run["model"]
    topo = run["topology"]
    corpus = run["corpus"]
    main = run["main_mode"]
    stages = run["modes"][main]
    other = next(mode for mode in run["modes"] if mode != main)
    eq = run["equivalence"]
    ladder = run["ladder"]

    unit_rows = "\n".join(
        f"| `{u['name']}` | {_n(u['numel'])} | {_n(u['padding'])} | {_n(u['padded'])} | "
        f"{_n(u['shard'])} |"
        for u in model["units"]
    )

    transient_rows = "\n".join(
        f"| {stages[k]['name']} | {_n(stages[k]['memory']['transient_peak']['gathered_params'])} | "
        f"{_n(stages[k]['memory']['transient_peak']['grad_bucket'])} |"
        for k in sorted(stages)
    )
    largest = stages["3"]["memory"]["largest_unit_bytes"]

    world_sizes = run["scaling"]["world_sizes"]
    by = {(r["stage"], r["world_size"]): r for r in run["scaling"]["rows"]}
    scaling_rows = []
    for stage in range(4):
        cells = []
        for n in world_sizes:
            row = by[(stage, n)]
            mark = "" if row["measured"] == row["predicted"] else " ✗"
            cells.append(f"{row['measured']:.4g}{mark}")
        scaling_rows.append(f"| {stages[str(stage)]['name']} | " + " | ".join(cells) + " |")
    scaling_head = "| stage | " + " | ".join(f"N = {n}" for n in world_sizes) + " |"
    scaling_rule = "| --- |" + " ---: |" * len(world_sizes)
    scaling_ok = all(r["measured"] == r["predicted"] for r in run["scaling"]["rows"])
    padded_rows = [r for r in run["scaling"]["rows"] if r["measured"] != r["measured_real"]]
    padding_note = (
        "Every entry is measured on bytes per **padded** element. Where padding is non-zero the "
        "figure per **real** weight is higher: "
        + "; ".join(
            f"N = {r['world_size']}, {stages[str(r['stage'])]['name']}: "
            f"{r['measured_real']:.6g} per real weight"
            for r in padded_rows
        )
        + "."
        if padded_rows
        else "Padding is zero at every N measured here."
    )

    comm_rows = []
    for key in sorted(stages):
        c = stages[key]["communication"]
        sent, pred = c["per_step_sent"], c["predicted"]
        multiple = comm_multiple(stages[key], topo["world_size"])
        comm_rows.append(
            f"| {stages[key]['name']} | {_n(sent.get('reduce_scatter', 0))} | "
            f"{_n(sent.get('all_gather', 0))} | **{_n(sent['total'])}** | {_n(pred['total'])} | "
            f"{_verdict(sent['total'], pred['total'])} | {multiple:g} × P·(N−1)/N | "
            f"{c['collectives_per_step']} |"
        )

    link_rows = "\n".join(
        f"| {stages[k]['name']} | "
        f"{_n(stages[k]['communication']['per_step_link_bytes']['intra'])} | "
        f"{_n(stages[k]['communication']['per_step_link_bytes']['inter'])} |"
        for k in sorted(stages)
    )
    compute_identical = all(
        stages[k]["compute"]["flops_ranks_identical"]
        and stages[k]["compute"]["optimizer_elements_ranks_identical"]
        for k in stages
    )
    key_bias_rows = "\n".join(
        [
            "| weights, largest absolute difference, key bias excluded | "
            f"{eq['fp32_weights_max_abs_except_key_bias']:.3g} |",
            f"| weights, largest absolute difference on the {eq['key_bias_elements']} key-bias "
            f"weights | {eq['fp32_weights_max_abs_key_bias']:.3g} |",
            "| the reference's own step-1 gradient on those key-bias weights | "
            f"{eq['key_bias_grad_step1_max_abs']:.3g} |",
        ]
    )

    compute_rows = []
    for key in sorted(stages):
        c = stages[key]["compute"]
        compute_rows.append(
            f"| {stages[key]['name']} | {_n(c['forward_flops'])} | {_n(c['recompute_flops'])} | "
            f"{_n(c['backward_flops'])} | {_n(c['optimizer_elements'])} | "
            f"{_fewer(optimizer_fewer(stages, key))} | {_n(c['optimizer_flops'])} |"
        )

    time_rows = "\n".join(
        f"| {stages[k]['name']} | {stages[k]['time_model']['compute_seconds'] * 1e6:.3f} | "
        f"{stages[k]['time_model']['comm_seconds'] * 1e6:.2f} | "
        f"{stages[k]['time_model']['total_seconds'] * 1e6:.2f} |"
        for k in sorted(stages)
    )

    equal_rows = "\n".join(
        f"| {mode} | "
        + " | ".join(f"{run['modes'][mode][k]['max_abs_vs_dp']:g}" for k in sorted(stages))
        + " |"
        for mode in (main, other)
    )

    ladder_ns = sorted({r["world_size"] for r in ladder["rows"]})
    lad = {(r["stage"], r["world_size"]): r for r in ladder["rows"]}
    ladder_rows = []
    for stage in range(4):
        cells = []
        for n in ladder_ns:
            r = lad[(stage, n)]
            cells.append(f"{r['gib']:.1f} {'fits' if r['fits'] else 'no'}")
        floor = lad[(stage, ladder_ns[0])]
        never = "never fits" if floor["never_fits"] else "can fit"
        ladder_rows.append(
            f"| {stages[str(stage)]['name']} | "
            + " | ".join(cells)
            + f" | {floor['floor_bytes'] / 2**30:.1f} GiB — {never} |"
        )
    ladder_head = "| stage | " + " | ".join(f"N = {n}" for n in ladder_ns) + " | floor (N → ∞) |"
    ladder_rule = "| --- |" + " ---: |" * len(ladder_ns) + " --- |"
    ladder_comm = "\n".join(
        f"| {stages[str(stage)]['name']} | "
        + " | ".join(
            f"{lad[(stage, n)]['comm_bytes_per_step'] / 1e9:.1f} GB · "
            f"{lad[(stage, n)]['comm_seconds_per_step']:.2f} s"
            for n in ladder_ns
        )
        + " |"
        for stage in range(4)
    )

    loss_rows = "\n".join(
        f"| {i + 1} | {stages['0']['losses'][i]:.6f} | "
        f"{run['modes']['fp32']['0']['losses'][i]:.6f} | {eq['reference_losses'][i]:.6f} |"
        for i in range(len(eq["reference_losses"]))
    )

    epochs_note = (
        "every sequence is read at most once"
        if corpus["epochs"] <= 1
        else "the run repeats text, so its losses measure memorisation"
    )
    transient_ok = all(
        stages[k]["memory"]["transient_peak"]["gathered_params"] <= largest["params"]
        and stages[k]["memory"]["transient_peak"]["grad_bucket"] <= largest["grads"]
        for k in stages
    )
    transient_verdict = (
        "no measured peak exceeds one unit"
        if transient_ok
        else "**a measured peak exceeds one unit**"
    )
    key_bias_verdict = (
        "Here the key-bias weights drifted further than any other weight, as that predicts."
        if key_bias_drifted_further(eq)
        else "Here the key-bias weights did **not** drift further than the rest."
    )
    env = prov["environment"]
    env_line = (
        f"python {env['python']} · torch {env['torch']} · {env['machine']} · "
        f"{env['torch_threads']} threads · {env['device']}"
    )
    return f"""# RESULTS — 12-distributed-training

> **Generated** by `tools/render_results.py` from `results/zero.json`, which
> `{run["generated_by"]}` wrote. Do not edit by hand: `tests/test_zerosim_results.py` fails if this
> file differs from a fresh render.

Every figure below is **measured** on the simulator unless its column or heading says
**predicted** (from the hand-derived formulas in `formulas.py`) or **time model** (assumed hardware
figures applied to measured counts — not a measurement of anything).

## Provenance

| field | value |
| --- | --- |
| config fingerprint | `{prov["config_fingerprint"]}` |
| code digest (zerosim + lossheads) | `{prov["code_digest"]}` |
| git commit | `{prov["git_sha"]}` |
| corpus digest | `{prov["corpus_digest"]}` |
| tokenizer digest | `{prov["tokenizer_digest"]}` |
| environment | {env_line} |

## The setup

**{topo["world_size"]} simulated devices** on {topo["nodes"]} nodes of {topo["devices_per_node"]}.
The ring visits them in rank order, so {topo["ring_links"]["intra"]} of its links stay inside a
node and {topo["ring_links"]["inter"]} cross between nodes.

**The model** is exercise 09's trunk ({cfg["model"]["n_layer"]} blocks, width
{cfg["model"]["d_model"]}, {cfg["model"]["n_head"]} heads) with an untied output head over a
{_n(cfg["model"]["vocab_size"])}-entry vocabulary: **{_n(model["params"])} weights**, split into
the units a ZeRO-3 device gathers one at a time. Padding makes each unit divide into
{topo["world_size"]} equal shards:

| unit | weights | padding | padded | shard per device |
| --- | ---: | ---: | ---: | ---: |
{unit_rows}

**Training.** {cfg["steps"]} AdamW steps (lr {cfg["learning_rate"]:g}, betas
{cfg["beta1"]:g}/{cfg["beta2"]:g}, weight decay {cfg["weight_decay"]:g}), one sequence of
{cfg["model"]["seq_len"]} tokens per device per step — a global batch of {topo["world_size"]}.
The text is exercise 09's frozen corpus: {_n(corpus["corpus_tokens"])} tokens, of which the run
reads {_n(corpus["tokens_consumed"])} ({corpus["epochs"]:.3f} epochs — {epochs_note}).

## 1 · Memory per device, measured by the ledger ({main})

Peak bytes held by **one** device in each persistent category, read from its ledger — the storage
bytes behind every tensor the stage put there, each storage counted once (a slice of a larger
buffer is charged the whole buffer). Activations and allocator overhead are not counted. The last
column compares every category against `formulas.bytes_per_device`.

{_memory_table(stages)}

Bytes per weight — measured total ÷ padded weights, against the hand formula
({FORMULA_TEXT[main]}):

{_bpw_table(stages)}

Every device's ledger is identical to device 0's in every stage:
{"yes" if all(stages[k]["memory"]["ranks_identical"] for k in stages) else "**NO**"}.

**The transient buffers** — the price of sharding, paid in short bursts. ZeRO-3 gathers one unit's
full weights for each forward and backward; ZeRO-2 and -3 hold one unit's full gradient between
its backward and its reduce-scatter. The largest unit is {_n(largest["params"])} bytes of weights
and {_n(largest["grads"])} bytes of gradient; {transient_verdict}:

| stage | gathered weights peak | gradient bucket peak |
| --- | ---: | ---: |
{transient_rows}

The same ledger in **{other}** ({FORMULA_TEXT[other]}):

{_memory_table(run["modes"][other])}

## 2 · Bytes per weight as N grows, measured ({main})

A fresh world at each N, one step each, ledger read afterwards. ✗ would mark a disagreement with
the formula; all agree: {"yes" if scaling_ok else "**NO**"}.

{scaling_head}
{scaling_rule}
{chr(10).join(scaling_rows)}

{padding_note}

## 3 · Communication per device per step, counted ({main})

Bytes **sent** by one device in one step, counted at every ring hop. `P` is one copy of the padded
weights in the gradient dtype: {_n(stages["0"]["communication"]["payload_bytes"])} bytes. A ring
pass is one reduce-scatter or one all-gather over one unit; an all-reduce is two.

| stage | reduce-scatter | all-gather | total | predicted | | in units of P | ring passes |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
{chr(10).join(comm_rows)}

Bytes carried per step by all links of each kind (summed over the ring):

| stage | intra-node links | inter-node links |
| --- | ---: | ---: |
{link_rows}

## 4 · Computation per device per step, counted ({main})

Forward and backward FLOPs are counted by `torch.utils.flop_counter` on each device (matrix
multiplies and attention; elementwise work is not counted). Every unit's forward runs twice — once
in the forward pass, once recomputed inside the backward (see `DECISIONS.md`) — identically in every
stage. The optimiser column is exact: how many weights this device's AdamW updated.

| stage | forward | recompute | backward | optimiser weights updated | vs DP | optimiser FLOPs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(compute_rows)}

Every device counted the same FLOPs and updated the same number of weights:
{"yes" if compute_identical else "**NO**"}.

## 5 · Time model — ASSUMED figures applied to the counts above

**Not a measurement.** Compute at an assumed {cfg["device_flops"] / 1e12:g} TFLOP/s per device;
links at an assumed {cfg["intra_node_bandwidth"] / 1e9:g} GB/s inside a node and
{cfg["inter_node_bandwidth"] / 1e9:g} GB/s between nodes; a collective lasts as long as its busiest
link; no overlap of compute and communication. Microseconds per step:

| stage | compute | communication | total |
| --- | ---: | ---: | ---: |
{time_rows}

## 6 · Does the stage change the answer?

**Largest absolute difference** in the fp32 weights the optimiser holds, after {cfg["steps"]} steps,
between each stage and data parallelism. `0` means bit-identical.

| precision | DP | ZeRO-1 | ZeRO-2 | ZeRO-3 |
| --- | ---: | ---: | ---: | ---: |
{equal_rows}

**Against one device trained on the whole global batch** with `torch.optim.AdamW` (fp32):

| check | measured |
| --- | ---: |
| loss, largest relative difference over all steps | {eq["fp32_loss_max_rel"]:.3g} |
| gradient after step 1, largest absolute difference | {eq["fp32_grad_step1_max_abs"]:.3g} |
| … the largest gradient it is measured against | {eq["fp32_grad_step1_scale"]:.3g} |
{key_bias_rows}

The key bias has a true gradient of exactly zero (it shifts a whole softmax row by a constant), so
what is computed for it is rounding noise, and AdamW scales every gradient by its own running size —
noise becomes a step. {key_bias_verdict}

**bf16-mixed against the same fp32 single device:**

| check | measured |
| --- | ---: |
| loss, largest relative difference | {eq["mixed_loss_max_rel"]:.3g} |
| weights, largest absolute difference | {eq["mixed_weights_max_abs"]:.3g} |
| cosine between the two runs' total weight updates | {eq["mixed_update_cosine"]:.6f} |
| root-mean-square size of the fp32 run's total update, for scale | {eq["update_rms"]:.3g} |

Mean loss per step:

| step | {main} (any stage) | fp32 (any stage) | one device, fp32 |
| ---: | ---: | ---: | ---: |
{loss_rows}

## 7 · The memory ladder — a {ladder["params"] / 1e9:g}B-weight model, PREDICTED

Nothing is simulated at this size. GiB per device from the formulas ({main}, activations excluded),
against an assumed card of {ladder["card_bytes"] / 1e9:g} GB ({ladder["card_gib"]:.1f} GiB). The
floor is what stays replicated however many devices are added.

{ladder_head}
{ladder_rule}
{chr(10).join(ladder_rows)}

Communication per device per step at that size, and its **time model** at the assumed link speeds
(a ring within one node of {topo["devices_per_node"]} runs at the intra-node speed; any larger ring
is paced by an inter-node link):

| stage | {" | ".join(f"N = {n}" for n in ladder_ns)} |
| --- |{" ---: |" * len(ladder_ns)}
{ladder_comm}
"""


#: The world size the page's ring figure is drawn at. A design choice for legibility — four
#: devices fit on a phone — and not a measurement; the page says so wherever it shows the figure.
RING_DEMO_WORLD_SIZE = 4

#: The world sizes the page's ladder slider can reach. `run["ladder"]` holds four of them; the
#: slider needs every integer in between, which `formulas.ladder` — the same function that wrote
#: those four — computes here. A prediction, labelled as one on the page.
LADDER_SLIDER_SIZES = tuple(range(1, 65))


def _sig(value: float, digits: int = 5) -> float:
    """A series value rounded to `digits` significant figures, for the page only."""
    return float(f"{value:.{digits}g}")


def _without_sources(value: object) -> object:
    """The bundle with every free-text `source` field removed.

    Those fields are notes for a reader of the JSON — which file the corpus came from, why a size
    was chosen. The page renders none of them, so they do not travel to it.
    """
    if isinstance(value, dict):
        return {k: _without_sources(v) for k, v in value.items() if k != "source"}
    if isinstance(value, list):
        return [_without_sources(v) for v in value]
    return value


def page_numbers(run: dict) -> dict:
    """Every number the page derives, computed once here rather than in the browser.

    The rule `docs/DESIGN.md` states twice: a generated table under a hand-written sentence looks
    maintained and only the sentence is wrong. So a ratio the page's prose quotes is a ratio this
    function returns, and the two multiples `RESULTS.md` also prints come from the same functions.
    """
    from zerosim import formulas

    main = run["main_mode"]
    world = run["topology"]["world_size"]
    padded = run["model"]["padded"]
    stages = run["modes"][main]
    categories = (*CATEGORIES, "total")

    per_weight = {
        mode: {
            key: {c: block["memory"]["measured"][c] / padded for c in categories}
            for key, block in blocks.items()
        }
        for mode, blocks in run["modes"].items()
    }
    transient_extra = {
        mode: {
            key: block["memory"]["peak_total"] - block["memory"]["measured"]["total"]
            for key, block in blocks.items()
        }
        for mode, blocks in run["modes"].items()
    }
    multiples = {
        mode: {key: comm_multiple(block, world) for key, block in blocks.items()}
        for mode, blocks in run["modes"].items()
    }
    # Stage 0 IS data parallelism, so its own difference from itself is no test of anything; the
    # comparisons worth counting are the ZeRO stages, in every precision.
    differences = [
        block["max_abs_vs_dp"]
        for blocks in run["modes"].values()
        for key, block in blocks.items()
        if key != "0"
    ]
    # Which of a weight's bytes each stage shards, and the replicated floor, straight from the
    # formulas the ledger was held to — so the page's per-stage diagram cannot disagree with them.
    formula = {
        mode: {
            key: {
                "floor": float(formulas.floor_bytes_per_weight(int(key), mode)),
                "sharded": {c: int(key) >= formulas.SHARDED_FROM[c] for c in CATEGORIES},
                # The formula's own value at the run's N, per category and in total, so the page
                # can set it beside the ledger's reading rather than print the ledger twice.
                "at_world": {
                    c: float(v) for c, v in formulas.bytes_per_weight(int(key), world, mode).items()
                },
                # Optimiser state sharded but gradients not: the device keeps its whole gradient
                # buffer, yet after the reduce-scatter only its own slice holds the summed
                # gradient, and only that slice is read by its update (stages._optimizer_view).
                "grad_slice_only": int(key) >= formulas.SHARDED_FROM["master"]
                and int(key) < formulas.SHARDED_FROM["grads"],
            }
            for key in blocks
        }
        for mode, blocks in run["modes"].items()
    }

    # The ring figure, at a world small enough to draw. Its chunk is a quarter of the real model's
    # gradient buffer, and the bytes per phase are formulas.comm_bytes_per_step — the function the
    # counters matched at N = 32 — so the figure states a prediction, not a count.
    n = RING_DEMO_WORLD_SIZE
    itemsize = stages["0"]["communication"]["payload_bytes"] // padded
    demo_padded = sum(-(-u["numel"] // n) * n for u in run["model"]["units"])
    demo_payload = demo_padded * itemsize
    if demo_payload % n:
        raise ValueError(f"a {demo_payload}-byte buffer does not cut into {n} equal chunks")
    phase = formulas.comm_bytes_per_step(0, n, demo_payload)["reduce_scatter"]
    if phase.denominator != 1:
        raise ValueError(f"one ring phase at N = {n} is a fractional byte count: {phase}")
    ring = {
        "world_size": n,
        "units": len(run["model"]["units"]),
        "padding": demo_padded - run["model"]["params"],
        "payload_bytes": demo_payload,
        "chunk_bytes": demo_payload // n,
        "phase_bytes": int(phase),
    }

    # The ladder at every N the slider reaches, from the same function as run["ladder"]. Exact
    # comparisons are made on the formula's own fractions before anything is rounded for display.
    ladder = run["ladder"]
    dense = formulas.ladder(ladder["params"], LADDER_SLIDER_SIZES, main, ladder["card_bytes"])
    for row in dense:
        published = next(
            (
                r
                for r in ladder["rows"]
                if (r["stage"], r["world_size"]) == (row["stage"], row["world_size"])
            ),
            None,
        )
        if published is not None and (published["bytes"], published["fits"]) != (
            row["bytes"],
            row["fits"],
        ):
            raise ValueError(f"the slider's ladder disagrees with the published one at {row}")
    first_fit = {}
    first_headroom = {}
    for stage in range(4):
        rows = [r for r in dense if r["stage"] == stage]
        fit = [r["world_size"] for r in rows if r["fits"]]
        room = [r["world_size"] for r in rows if r["bytes"] < ladder["card_bytes"]]
        first_fit[str(stage)] = fit[0] if fit else None
        first_headroom[str(stage)] = room[0] if room else None
    slider = {
        "world_sizes": list(LADDER_SLIDER_SIZES),
        "gib": {
            str(stage): [_sig(r["gib"]) for r in dense if r["stage"] == stage] for stage in range(4)
        },
        "fits": {
            str(stage): [r["fits"] for r in dense if r["stage"] == stage] for stage in range(4)
        },
        "floor_gib": {
            str(stage): next(r["floor_bytes"] for r in dense if r["stage"] == stage) / 2**30
            for stage in range(4)
        },
        "never_fits": {
            str(stage): next(r["never_fits"] for r in dense if r["stage"] == stage)
            for stage in range(4)
        },
        "first_fit": first_fit,
        "first_headroom": first_headroom,
        # True where the first size that fits fills the card to the byte, leaving nothing for
        # activations — so the page may say so only when it is so.
        "exact_fill": {
            k: first_fit[k] is not None and first_headroom[k] != first_fit[k] for k in first_fit
        },
    }

    return {
        "per_weight": per_weight,
        "transient_extra": transient_extra,
        "comm_multiple": multiples,
        "optimizer_fewer": {key: optimizer_fewer(stages, key) for key in stages},
        "zero3_sent_over_dp": stages["3"]["communication"]["per_step_sent"]["total"]
        / stages["0"]["communication"]["per_step_sent"]["total"],
        "dp_bytes_over_zero3": stages["0"]["memory"]["measured"]["total"]
        / stages["3"]["memory"]["measured"]["total"],
        "max_abs_vs_dp": max(differences),
        "zero_runs_compared": len(differences),
        # The verdict words the page prints, decided here and not in the browser.
        "memory_equal": {
            mode: {
                key: {
                    c: block["memory"]["measured"][c] == block["memory"]["predicted"][c]
                    for c in categories
                }
                for key, block in blocks.items()
            }
            for mode, blocks in run["modes"].items()
        },
        "scaling_all_match": all(r["measured"] == r["predicted"] for r in run["scaling"]["rows"]),
        "identical_to_dp": {
            mode: {key: block["max_abs_vs_dp"] == 0 for key, block in blocks.items()}
            for mode, blocks in run["modes"].items()
        },
        "formula": formula,
        "key_bias_drifted_further": key_bias_drifted_further(run["equivalence"]),
        # One run, one seed, one bare `>`: the ratio is reported so the page can say how far apart
        # the two maxima are, and that no noise floor was measured for it.
        "key_bias_ratio": (
            run["equivalence"]["fp32_weights_max_abs_key_bias"]
            / run["equivalence"]["fp32_weights_max_abs_except_key_bias"]
            if run["equivalence"]["fp32_weights_max_abs_except_key_bias"]
            else None
        ),
        "reference_differs": run["equivalence"]["fp32_weights_max_abs_except_key_bias"] > 0
        or run["equivalence"]["fp32_weights_max_abs_key_bias"] > 0,
        "all_memory_equal": all(
            block["memory"]["measured"][c] == block["memory"]["predicted"][c]
            for blocks in run["modes"].values()
            for block in blocks.values()
            for c in categories
        ),
        "all_comm_equal": all(
            block["communication"]["per_step_sent"]["total"]
            == block["communication"]["predicted"]["total"]
            for blocks in run["modes"].values()
            for block in blocks.values()
        ),
        "all_ranks_identical": all(
            block["memory"]["ranks_identical"] and block["communication"]["ranks_identical"]
            for blocks in run["modes"].values()
            for block in blocks.values()
        )
        and all(
            block["compute"]["flops_ranks_identical"]
            and block["compute"]["optimizer_elements_ranks_identical"]
            for block in stages.values()
        ),
        "all_identical_to_dp": all(d == 0 for d in differences),
        "zero12_send_same_as_dp": all(
            stages[k]["communication"]["per_step_sent"]["total"]
            == stages["0"]["communication"]["per_step_sent"]["total"]
            for k in ("1", "2")
        ),
        "padded_world_sizes": sorted(
            {r["world_size"] for r in run["scaling"]["rows"] if r["measured"] != r["measured_real"]}
        ),
        # Whether a stage's two kinds of transient buffer were held at the same moment: its peak
        # above the persistent total is their sum if so, the larger alone if not.
        "transient_together": {
            mode: {
                key: min(block["memory"]["transient_peak"].values()) > 0
                and block["memory"]["peak_total"] - block["memory"]["measured"]["total"]
                == sum(block["memory"]["transient_peak"].values())
                for key, block in blocks.items()
            }
            for mode, blocks in run["modes"].items()
        },
        "passes_per_unit": {
            key: block["communication"]["collectives_per_step"] / len(run["model"]["units"])
            for key, block in stages.items()
        },
        "ring": ring,
        "slider": slider,
    }


def render_page_data(run: dict) -> str:
    """Generate `web/data.js` — every figure the page draws, read from the same bundle.

    **The page must not hold a number of its own.** `chapters.js` reads `M.*` and writes nothing;
    this file is regenerated by `main()` and checked against a fresh render by
    `tests/test_zerosim_results.py`. It needs no `torch`: `formulas` is pure arithmetic.
    """
    page = _without_sources(run)
    page["page"] = page_numbers(run)
    return (
        "/* GENERATED by tools/render_results.py. Do not edit.\n"
        " *\n"
        " * Every number the page draws is in here, read from results/zero.json or computed\n"
        " * from it by zerosim.formulas. `chapters.js` holds none of its own.\n"
        " */\n"
        "export const M = " + json.dumps(page, indent=2, sort_keys=True) + ";\n"
    )


def main() -> int:
    """Render `results/zero.json` into `RESULTS.md`, and into `web/data.js` when the page exists."""
    run = json.loads(RESULTS.read_text(encoding="utf-8"))
    OUT.write_text(render(run), encoding="utf-8")
    print(f"wrote {OUT}")
    page_data = EXERCISE / "web" / "data.js"
    if page_data.parent.is_dir():
        page_data.write_text(render_page_data(run), encoding="utf-8")
        print(f"wrote web/{page_data.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
