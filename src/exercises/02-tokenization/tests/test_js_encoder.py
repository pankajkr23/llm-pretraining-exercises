"""The JavaScript encoder must produce the same tokens as Python, character for character.

The widget ships a merge list and an encoder so a reviewer can reproduce our counts. That promise
is only worth something if the two implementations agree, and the place they drift is always the
pre-tokenizer — a different whitespace rule, a leading marker added or not, a newline kept or
dropped. So this runs real corpus lines through both and compares token streams exactly.

Skips (rather than fails) when node or ``web/data.json`` is absent, so a fresh checkout without
node still has a working suite — and so that the skip is visible in the run output instead of
looking like a pass.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from tokenization.ablate import train_spec
from tokenization.config import PROFILES, Config
from tokenization.corpus import load_all
from tokenization.widget import FEATURED

WEB = Path(__file__).resolve().parents[1] / "web"
DATA = WEB / "data.json"

# Lines chosen to hit the pre-tokenizer's edge cases rather than average prose: a literal
# underscore (which must NOT be confused with the U+2581 marker), markdown link machinery, digits
# with separators, every script in the corpus, and runs of whitespace.
PROBES = [
    "India is a country in South Asia.",
    "snake_case_name and __dunder__ stay intact",
    "[India](https://en.wikipedia.org/wiki/India) · population 1,428,627,663.",
    "भारत एक देश है। भारत की राजधानी नई दिल्ली है।",
    "భారతదేశం ఒక దేశం. తెలుగు ఒక భాష.",
    "  leading and   internal   runs of spaces  ",
    "trailing punctuation!!! ...and — dashes",
]

# Compare **ids**, not token strings. The three engines disagree about what to *call* an unknown
# symbol — HuggingFace reports the unknown token, our from-scratch BPE reports the original
# character, and the JS keeps the character so the page can show it in a chip — while all three
# agree on its id. Ids are also the thing that actually reproduces a score, since the score is a
# count of them, so this compares what matters rather than three spellings of the same fact.
_RUNNER = """
import { encode } from '%s';
import { readFileSync } from 'node:fs';
const data = JSON.parse(readFileSync(%s, 'utf8'));
const config = data.configs.find((c) => c.label === %s);
const probes = JSON.parse(readFileSync(%s, 'utf8'));
console.log(JSON.stringify(probes.map((p) => encode(p, config).map((t) => t.id))));
"""


def _node() -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed — JS/Python parity unchecked")
    return node


@pytest.fixture(scope="module")
def payload() -> dict:
    if not DATA.exists():
        pytest.skip("web/data.json not built — run `uv run python -m tokenization.widget`")
    return json.loads(DATA.read_text(encoding="utf-8"))


def _js_tokens(node: str, tmp_path: Path, label: str, probes: list[str]) -> list[list[str]]:
    probe_file = tmp_path / "probes.json"
    probe_file.write_text(json.dumps(probes, ensure_ascii=False), encoding="utf-8")
    script = tmp_path / "run.mjs"
    script.write_text(
        _RUNNER
        % (
            (WEB / "encoder.js").as_uri(),
            json.dumps(str(DATA)),
            json.dumps(label),
            json.dumps(str(probe_file)),
        ),
        encoding="utf-8",
    )
    out = subprocess.run(  # noqa: S603 - fixed argv, paths we wrote ourselves
        [node, str(script)], capture_output=True, text=True, check=True, timeout=120
    )
    return json.loads(out.stdout)


@pytest.mark.integration
@pytest.mark.parametrize("spec", [s for s in FEATURED if s.algo in {"bpe", "bpe-scratch"}])
def test_js_encoder_matches_python_token_for_token(spec, payload, tmp_path):
    """Both profiles, because they pre-tokenize differently.

    v1's tokenizers use HuggingFace's default Metaspace ``prepend_scheme="always"`` and v2's pin
    it to ``"never"`` — a one-character difference in the first pre-token that the JS has to
    honour, and would otherwise silently get wrong for half the page.
    """
    node = _node()
    config = next(c for c in payload["configs"] if c["label"] == spec.label)
    if config["encoder"]["kind"] == "unsupported":
        pytest.skip(f"no JS encoder for {config['encoder']['reason']}")

    cfg = Config()
    tok = train_spec(spec, load_all(PROFILES[spec.profile], cfg.corpus_dir))

    js = _js_tokens(node, tmp_path, spec.label, PROBES)
    for probe, js_ids in zip(PROBES, js, strict=True):
        assert tok.encode(probe).ids == js_ids, f"diverged on {probe!r} ({spec.profile})"


@pytest.mark.integration
def test_cross_check_would_catch_a_broken_pre_tokenizer(payload, tmp_path):
    # Prove the comparison has teeth: feed the JS encoder a config whose pre-tokenizer rule has
    # been swapped, and the token streams must stop matching.
    node = _node()
    config = next((c for c in payload["configs"] if c["encoder"]["kind"] == "metaspace-bpe"), None)
    if config is None:
        pytest.skip("no metaspace BPE config exported")

    broken = json.loads(json.dumps(config))
    broken["encoder"]["kind"] = "scratch-bpe"  # splits on all whitespace, drops newlines
    broken["label"] = "broken"
    payload_path = tmp_path / "data.json"
    payload_path.write_text(
        json.dumps({"configs": [config, broken]}, ensure_ascii=False), encoding="utf-8"
    )

    probes = ["a b\nc", "  spaced  out  "]
    probe_file = tmp_path / "probes.json"
    probe_file.write_text(json.dumps(probes, ensure_ascii=False), encoding="utf-8")

    def run(label: str) -> list[list[str]]:
        script = tmp_path / f"run-{label}.mjs"
        script.write_text(
            _RUNNER
            % (
                (WEB / "encoder.js").as_uri(),
                json.dumps(str(payload_path)),
                json.dumps(label),
                json.dumps(str(probe_file)),
            ),
            encoding="utf-8",
        )
        out = subprocess.run(  # noqa: S603 - fixed argv, paths we wrote ourselves
            [node, str(script)], capture_output=True, text=True, check=True, timeout=120
        )
        return json.loads(out.stdout)

    assert run(config["label"]) != run("broken")
