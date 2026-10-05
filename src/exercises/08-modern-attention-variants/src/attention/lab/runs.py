"""What produced a lab result, and the refusal to write one that cannot say.

`AGENTS.md` requires every script that produces a number a document renders to write a bundle
carrying *which settings*, *which code*, *which commit*, *which machine* and *which data* — and to
refuse, not warn, when any is missing. The six fields here are the same six exercise 09 records,
and most of them are exercise 09's own functions, reused rather than copied:

| field | from |
| --- | --- |
| `config_fingerprint` | exercise 09's `config_fingerprint`, on a dataclass whose `repr` is stable |
| `code_digest` | **ours** — see `code_digest` for why exercise 09's cannot be reused |
| `git_sha` | exercise 09's `git_sha` |
| `environment` | exercise 09's `environment`, plus numpy's version and `describe_device` |
| `corpus_digest` | exercise 09's `corpus_digest`, for a task that reads the frozen corpus |
| `tokenizer_digest` | exercise 09's `tokenizer_digest`, likewise |

A task that never reads the corpus (the synthetic recall test, the cost measurement) records a
digest of the exact tensors it generated as `corpus_digest`, and says in `tokenizer_digest` that no
tokenizer was used. Recording the frozen corpus's digest there instead would name an input the
numbers do not depend on.

**Results are written under this exercise's `artifacts/` and nowhere else.** `save` refuses any
other path, `results/` included: a lab run is exploratory until someone decides a document should
render it, and promoting it to `results/` is that decision, made by a person.
"""

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any

import torch

LAB_DIR = Path(__file__).resolve().parent
"""This package's directory — `attention/lab/`."""

ATTENTION_DIR = LAB_DIR.parent
"""The `attention` package, whose `catalogue` and `sources` modules the lab imports."""

EXERCISE = LAB_DIR.parents[2]
"""Exercise 08's root — the directory holding `src/`, `results/` and `artifacts/`."""

ARTIFACTS = EXERCISE / "artifacts"
"""The only directory `save` writes under. Read at call time, so a test can point it elsewhere."""

RESULTS_SUBDIR = "lab"
"""`save`'s default subdirectory of `ARTIFACTS`."""

REQUIRED_FIELDS = (
    "config_fingerprint",
    "code_digest",
    "git_sha",
    "corpus_digest",
    "tokenizer_digest",
    "environment",
)
"""Every field `save` insists on — the same six as exercise 09's `REQUIRED_FIELDS`."""

#: The `attention` modules outside `lab/` that the lab imports: `hparams` reads the catalogue,
#: and the catalogue builds its `Source` records. A change there can change a parameter value.
ATTENTION_MODULES = ("catalogue.py", "sources.py")


# --- which settings -------------------------------------------------------------------------------


def config_fingerprint(config: Any) -> str:
    """Exercise 09's fingerprint — `blake2b` over the sorted fields — of a dataclass.

    Exercise 09's function is annotated for its own `Config` but only needs `asdict`, so any
    dataclass works. What it cannot tolerate is a field whose `repr` names a memory address — a
    function or a module object — because that changes on every run and the fingerprint would
    never match itself. That is refused here rather than discovered as two "different" runs.

    Raises:
        TypeError: When `config` is not a dataclass instance.
        ValueError: When its `repr` holds an object address.
    """
    from dataclasses import asdict

    from lossheads.provenance import config_fingerprint as fingerprint

    if not is_dataclass(config) or isinstance(config, type):
        raise TypeError(f"config_fingerprint needs a dataclass instance, got {type(config)}")
    if " at 0x" in repr(sorted(asdict(config).items())):
        raise ValueError(
            "this configuration holds an object whose repr is a memory address, so its "
            "fingerprint would differ on every run; describe the object by name instead"
        )
    return fingerprint(config)


# --- which code -----------------------------------------------------------------------------------


def code_files() -> list[tuple[str, Path]]:
    """`(label, path)` for every module a lab number depends on, sorted by label.

    Every `attention/lab/*.py`, the two `attention` modules the lab imports, and every
    `lossheads/*.py` — exercise 09's package, which supplies the corpus, the tokenizer and the
    provenance helpers.
    """
    import lossheads

    lossheads_dir = Path(lossheads.__file__).resolve().parent
    files = [(f"attention/lab/{p.name}", p) for p in LAB_DIR.glob("*.py")]
    files += [(f"attention/{name}", ATTENTION_DIR / name) for name in ATTENTION_MODULES]
    files += [(f"lossheads/{p.name}", p) for p in lossheads_dir.glob("*.py")]
    return sorted(files)


def code_digest(files: Sequence[tuple[str, Path]] | None = None) -> str:
    """`sha256:<64 hex>` over every module the numbers depend on, in label order.

    **Exercise 09's `code_digest` cannot be reused**, and not by preference: it globs the directory
    of its own file (`Path(__file__).parent`) and nothing else, so called from here it would digest
    exercise 09 and vouch for a lab it never read. This one covers both packages. Each file
    contributes its label and its bytes, so renaming a module moves the digest as well as editing
    one.

    Args:
        files: `(label, path)` pairs; defaults to `code_files()`. Passing a list is how a test
            checks the digest follows the bytes without editing a real file.
    """
    digest = hashlib.sha256()
    for name, path in sorted(files if files is not None else code_files()):
        digest.update(name.encode("utf-8"))
        digest.update(Path(path).read_bytes())
    return "sha256:" + digest.hexdigest()


# --- which data -----------------------------------------------------------------------------------


def tensor_digest(tensors: Iterable[torch.Tensor]) -> str:
    """`sha256:<64 hex>` over tensors' dtypes, shapes and bytes, in order."""
    digest = hashlib.sha256()
    for tensor in tensors:
        tensor = tensor.detach().to("cpu").contiguous()
        digest.update(f"{tensor.dtype}{tuple(tensor.shape)}".encode())
        digest.update(tensor.numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def combine_digests(digests: Iterable[str]) -> str:
    """One `sha256:` digest standing for several, order-sensitive."""
    return "sha256:" + hashlib.sha256("\n".join(digests).encode("utf-8")).hexdigest()


# --- which machine --------------------------------------------------------------------------------


def select_device(requested: str | None = None) -> torch.device:
    """The device to run on: what was asked for, or the fastest available.

    Exercise 07's order — CUDA, then Apple's MPS, then CPU — copied rather than imported, so the
    lab does not depend on exercise 07.
    """
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device: torch.device) -> dict[str, object]:
    """What ran, and whether MPS was built but unavailable — exercise 07's pattern.

    A sandbox that blocks the OS-version query makes `torch.backends.mps.is_available()` return
    `False`, and automatic selection then falls back to the CPU without a word. On Apple silicon,
    `mps_built` without `mps_available` is that trap's signature, so it is recorded as a field.
    """
    import platform

    built = bool(torch.backends.mps.is_built())
    available = bool(torch.backends.mps.is_available())
    apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    return {
        "device": device.type,
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_built": built,
        "mps_available": available,
        "mps_unavailable_but_built": apple_silicon and built and not available,
    }


def environment(device: torch.device | str | None = None) -> dict[str, Any]:
    """Exercise 09's environment block, plus numpy's version and the device description."""
    import numpy
    from lossheads.provenance import environment as base_environment

    device = torch.device(device or "cpu")
    return base_environment(str(device)) | {"numpy": numpy.__version__} | describe_device(device)


# --- the block, and the refusal -------------------------------------------------------------------


def provenance(
    config: Any,
    device: torch.device | str | None = None,
    data_digests: dict[str, str] | None = None,
) -> dict[str, Any]:
    """The six-field block for a bundle's `"provenance"` key.

    Args:
        config: The dataclass the run was configured by.
        device: Where it ran.
        data_digests: `corpus_digest` and `tokenizer_digest` for a task that does not read the
            frozen corpus. `None` records the frozen corpus's and tokenizer's own digests.
    """
    from lossheads import provenance as base

    if data_digests is None:
        data = {"corpus_digest": base.corpus_digest(), "tokenizer_digest": base.tokenizer_digest()}
    else:
        data = {k: data_digests.get(k, "") for k in ("corpus_digest", "tokenizer_digest")}
    return {
        "config_fingerprint": config_fingerprint(config),
        "code_digest": code_digest(),
        "git_sha": base.git_sha(),
        "environment": environment(device),
        **data,
    }


def _empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (dict, list, tuple, set)):
        return not value
    return False


def missing_fields(bundle: dict[str, Any]) -> list[str]:
    """The required provenance fields that are absent, `None`, blank or empty."""
    block = bundle.get("provenance")
    if not isinstance(block, dict):
        return list(REQUIRED_FIELDS)
    return [name for name in REQUIRED_FIELDS if _empty(block.get(name))]


def _plain(value: Any) -> Any:
    """JSON-ready data: a non-finite float becomes its name, anything unencodable is an error."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(
        f"a {type(value).__name__} cannot be written to a result bundle; convert it to plain "
        "data first (a device is its .type, a tensor is .tolist())"
    )


def _destination(path: Path | str | None, bundle: dict[str, Any]) -> Path:
    root = Path(ARTIFACTS).resolve()
    if path is None:
        fingerprint = bundle["provenance"]["config_fingerprint"]
        task = str(bundle.get("task", "run"))
        path = root / RESULTS_SUBDIR / f"{task}-{fingerprint}.json"
    target = Path(path).resolve()
    if not target.is_relative_to(root) or target == root:
        raise ValueError(
            f"refusing to write {target}: lab results go under {root} and nowhere else. "
            "Promoting a run to results/ is a person's decision, not this function's."
        )
    if target.suffix != ".json":
        raise ValueError(f"refusing to write {target}: a result bundle is a .json file")
    return target


def save(bundle: dict[str, Any], path: Path | str | None = None) -> Path:
    """Write `bundle` as JSON under `artifacts/`, refusing one that cannot say where it came from.

    The whole bundle is encoded before the file is opened, so a refusal or an encoding error
    leaves nothing half-written.

    Args:
        bundle: Must carry a `"provenance"` block with every field in `REQUIRED_FIELDS`, none
            of them empty.
        path: Defaults to `artifacts/lab/<task>-<config_fingerprint>.json`.

    Returns:
        The path written.

    Raises:
        ValueError: For a missing or empty provenance field, a path outside `artifacts/`, a
            non-JSON path, or a value JSON cannot hold.
    """
    missing = missing_fields(bundle)
    if missing:
        raise ValueError(
            "refusing to write a result that cannot say where it came from; missing or empty: "
            f"{', '.join(missing)}. A number nobody can regenerate is not evidence."
        )
    target = _destination(path, bundle)
    text = json.dumps(_plain(bundle), indent=2, allow_nan=False) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def load(path: Path | str) -> dict[str, Any]:
    """Read a bundle `save` wrote."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
