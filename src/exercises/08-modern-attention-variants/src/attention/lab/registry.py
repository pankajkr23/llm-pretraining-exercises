"""Where variants are registered, and where every consumer finds them.

Adding a variant is one `register(MixerSpec(...))` in its family module. The notebook, the
documentation generator and the generic tests all iterate `specs()`, so nothing else changes.

`load_all()` imports every family module, which is what performs the registrations. It is explicit
rather than a directory scan: a module that fails to import must fail loudly, not vanish from the
list — `tests/test_attention_lab.py` separately checks that every catalogue mechanism is covered.
"""

import importlib

from attention.lab.base import FAMILIES, MixerSpec

#: The family modules, in the order the documentation and the notebook present them.
MODULES = (
    "core",
    "positions",
    "mla",
    "sparse",
    "linear",
    "delta",
    "ssm",
    "hybrid",
)

_SPECS: dict[str, MixerSpec] = {}


def register(spec: MixerSpec) -> MixerSpec:
    """Add a variant. A second registration under the same name is an error, never an overwrite."""
    if spec.name in _SPECS:
        raise ValueError(f"variant {spec.name!r} is registered twice")
    _SPECS[spec.name] = spec
    return spec


def load_all() -> None:
    """Import every family module so their registrations run."""
    for module in MODULES:
        importlib.import_module(f"attention.lab.{module}")


def get(name: str) -> MixerSpec:
    """The variant called `name`."""
    load_all()
    try:
        return _SPECS[name]
    except KeyError:
        raise KeyError(f"no variant {name!r}; known: {sorted(_SPECS)}") from None


def specs(family: str | None = None) -> list[MixerSpec]:
    """Every registered variant, in family order then registration order."""
    load_all()
    if family is not None and family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    ordered = sorted(_SPECS.values(), key=lambda s: FAMILIES.index(s.family))
    return [s for s in ordered if family is None or s.family == family]


def names(family: str | None = None) -> list[str]:
    """Registry names, in the same order as `specs`."""
    return [s.name for s in specs(family)]
