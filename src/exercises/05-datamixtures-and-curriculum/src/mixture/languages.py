"""Which Indic languages enter the run, and when — decided by measurement, not by preference.

The session asks this directly. From the transcript, stating the assignment in his own words:

    *"...reason when I going to train on Sanskrit if ever or urdu or other languages — right, so
    you need to come up with everything now."*

A plan that answers with a single "Indic 18%" has not answered it. `lanes.py` splits that 18% by
**provenance**; this module splits it by **language and time**, which is the other half.

**The gate is measured, and it is brutal.** Every South Asian language in FLORES-200 was tokenised
with our own Session 2 vocabulary. Exercise 04's rule is that a count which is mostly `[UNK]` is
not a count; applied to training text, a language above the gate cannot be trained on at all,
because the model would be fitting the unknown-token id rather than the language.

The result is a clean split along **script**, not language:

- **Readable** (≤5% `[UNK]`): English, Telugu, and every Devanagari language measured — Hindi,
  Marathi, Nepali, Bhojpuri, Magahi, Maithili, Chhattisgarhi, Kashmiri-in-Devanagari, **Sanskrit**.
- **Unreachable** (77–88% `[UNK]`): everything in any other script. **Urdu**, Sindhi and
  Kashmiri-in-Perso-Arabic; Bengali, Assamese and Manipuri-in-Bengali; Punjabi, Gujarati, Odia,
  Sinhala, Santali, Kannada, Tamil, Malayalam.

So the honest answers to the two languages named in the session are **Sanskrit: yes, and it can
start early — 0.1% `[UNK]`, because it shares Devanagari with Hindi. Urdu: not until the
vocabulary is retrained — 77.7%.** Nine Devanagari languages arrived free with Hindi; fourteen
languages are shut out by script alone. That single measurement is the strongest argument in
`TOKENIZER.md` for a larger vocabulary, and it is an argument this exercise made by measuring
rather than asserting.

**FLORES-200 is gitignored**, so the measured table is cached below and
`measure_readability()` recomputes it wherever the data is present. A test compares the two and
fails on drift, which is what keeps a cached number from becoming folklore.
"""

from dataclasses import dataclass

from mixture.config import Config

# Exercise 04's publication gate, reused as an admission gate. Above this, a language is not text
# the model can learn -- it is a stream of unknown-token ids wearing a language's name.
UNK_GATE = 0.05

# Measured with `ours/s02-bpe-10000` over the first 400 lines of each FLORES-200 dev file.
# Regenerate with `uv run python -m mixture.languages`.
MEASURED: dict[str, tuple[float, float]] = {
    # code: ([UNK] share, tokens per word)
    "eng_Latn": (0.000, 2.05),
    "hin_Deva": (0.000, 2.13),
    "bho_Deva": (0.000, 2.29),
    "hne_Deva": (0.000, 2.41),
    "mag_Deva": (0.001, 2.43),
    "mai_Deva": (0.000, 2.56),
    "kas_Deva": (0.000, 2.94),
    "npi_Deva": (0.000, 3.36),
    "mar_Deva": (0.006, 3.63),
    "tel_Telu": (0.002, 3.77),
    "san_Deva": (0.001, 4.00),
    "urd_Arab": (0.777, 4.64),
    "snd_Arab": (0.776, 4.66),
    "pan_Guru": (0.781, 5.14),
    "kas_Arab": (0.804, 5.51),
    "sat_Olck": (0.816, 5.81),
    "guj_Gujr": (0.816, 6.03),
    "sin_Sinh": (0.813, 6.26),
    "asm_Beng": (0.821, 6.66),
    "ben_Beng": (0.826, 6.69),
    "ory_Orya": (0.829, 6.81),
    "mni_Beng": (0.839, 7.18),
    "kan_Knda": (0.852, 8.36),
    "tam_Taml": (0.869, 8.96),
    "mal_Mlym": (0.877, 9.60),
}

NAMES: dict[str, str] = {
    "eng_Latn": "English",
    "hin_Deva": "Hindi",
    "bho_Deva": "Bhojpuri",
    "hne_Deva": "Chhattisgarhi",
    "mag_Deva": "Magahi",
    "mai_Deva": "Maithili",
    "kas_Deva": "Kashmiri (Devanagari)",
    "npi_Deva": "Nepali",
    "mar_Deva": "Marathi",
    "tel_Telu": "Telugu",
    "san_Deva": "Sanskrit",
    "urd_Arab": "Urdu",
    "snd_Arab": "Sindhi",
    "pan_Guru": "Punjabi",
    "kas_Arab": "Kashmiri (Perso-Arabic)",
    "sat_Olck": "Santali",
    "guj_Gujr": "Gujarati",
    "sin_Sinh": "Sinhala",
    "asm_Beng": "Assamese",
    "ben_Beng": "Bengali",
    "ory_Orya": "Odia",
    "mni_Beng": "Manipuri",
    "kan_Knda": "Kannada",
    "tam_Taml": "Tamil",
    "mal_Mlym": "Malayalam",
}


@dataclass(frozen=True)
class LanguagePlan:
    """When one language enters the run, and why then.

    Attributes:
        code: FLORES-200 code, which carries the script.
        name: The language.
        script: Writing system, taken from the code's suffix.
        unk: Measured `[UNK]` share under our vocabulary.
        fertility: Measured tokens per word under our vocabulary.
        readable: Whether it clears `UNK_GATE`.
        wave: When it enters — a stage key, or `blocked` where it cannot.
        share_of_indic: Its share of the Indic lane, or 0 while blocked.
        because: The reason, in the terms a reviewer will push on.
    """

    code: str
    name: str
    script: str
    unk: float
    fertility: float
    readable: bool
    wave: str
    share_of_indic: float
    because: str


# Waves. The rule is deliberately simple, because a complicated one would be unfalsifiable:
#
#   1. `seed`      -- readable AND the inventory holds verified-native text for it.
#   2. `general`   -- readable, arriving free with the script, but thinner on verified text.
#   3. `blocked`   -- above the gate. Not scheduled at all until the vocabulary is retrained;
#                     writing a share for a language the model cannot encode is the wishful
#                     accounting this exercise argues against, applied to languages.
#
# Shares are of the Indic lane and are set by supply and by fertility: a language that costs 4.00
# tokens per word buys less per token than one at 2.13, so an equal split would quietly spend more
# of the budget on the languages that compress worst.
_SEED = {"hin_Deva": 0.42, "tel_Telu": 0.20, "mai_Deva": 0.06}
_GENERAL = {
    "mar_Deva": 0.12,
    "npi_Deva": 0.07,
    "bho_Deva": 0.05,
    "mag_Deva": 0.03,
    "hne_Deva": 0.03,
    "kas_Deva": 0.01,
    "san_Deva": 0.01,
}

_WHY = {
    "hin_Deva": "the largest verified-native pool in Sangraha and the language MILU weights most",
    "tel_Telu": "the only non-Devanagari script the vocabulary reads, and exercise 04's Indic "
    "corpus is Devanagari and Telugu for exactly that reason",
    "mai_Deva": "carried from Session 2, where it was up-weighted x3 because it shares Devanagari "
    "with Hindi and won almost no merges of its own",
    "mar_Deva": "readable at 0.6% and the largest Devanagari language after Hindi",
    "npi_Deva": "readable at 0.0%; arrives free with the script",
    "bho_Deva": "readable at 0.0%; arrives free with the script",
    "mag_Deva": "readable at 0.1%; arrives free with the script",
    "hne_Deva": "readable at 0.0%; arrives free with the script",
    "kas_Deva": "readable at 0.0% in Devanagari, while the same language in Perso-Arabic is at "
    "80.4% -- the clearest evidence that the gate is about script, not language",
    "san_Deva": "**the session asks about this one by name.** Readable at 0.1% because it is "
    "Devanagari, so it can enter; held to 1% because its supply is thin and its "
    "fertility is the worst of the readable set at 4.00 tokens per word",
}

_BLOCKED_WHY = (
    "above the {gate:.0%} gate at {unk:.0%} `[UNK]`. The vocabulary has no merges for {script}, so "
    "training on it would fit the unknown-token id rather than the language. Unblocked only by the "
    "retokenisation TOKENIZER.md argues for"
)


def plan() -> tuple[LanguagePlan, ...]:
    """The per-language schedule, ordered by wave and then by share.

    Returns:
        One entry per measured South Asian language, English excluded -- it is the web lane's
        business, not the Indic lane's.
    """
    out: list[LanguagePlan] = []
    for code, (unk, fertility) in MEASURED.items():
        if code == "eng_Latn":
            continue
        script = code.split("_")[1]
        readable = unk <= UNK_GATE
        if code in _SEED:
            wave, share = "seed", _SEED[code]
        elif code in _GENERAL:
            wave, share = "general", _GENERAL[code]
        else:
            wave, share = "blocked", 0.0
        because = (
            _WHY.get(code, "readable, but not scheduled in this plan")
            if readable
            else _BLOCKED_WHY.format(gate=UNK_GATE, unk=unk, script=script)
        )
        out.append(
            LanguagePlan(
                code, NAMES.get(code, code), script, unk, fertility, readable, wave, share, because
            )
        )
    order = {"seed": 0, "general": 1, "blocked": 2}
    return tuple(sorted(out, key=lambda p: (order[p.wave], -p.share_of_indic, p.code)))


def scheduled() -> tuple[LanguagePlan, ...]:
    """Languages that actually receive tokens.

    Returns:
        Those not blocked.
    """
    return tuple(p for p in plan() if p.wave != "blocked")


def blocked() -> tuple[LanguagePlan, ...]:
    """Languages the vocabulary cannot encode.

    Returns:
        Those above the gate.
    """
    return tuple(p for p in plan() if p.wave == "blocked")


def shares_sum() -> float:
    """Total of the per-language shares of the Indic lane.

    Returns:
        Sum over scheduled languages. `checks.py` requires this to be 1.
    """
    return sum(p.share_of_indic for p in scheduled())


def tokens(code: str, config: Config | None = None) -> float:
    """Tokens one language receives at the configured run size.

    Args:
        code: FLORES-200 code.
        config: Thresholds and run size; defaults to `Config()`.

    Returns:
        Its share of the Indic lane times that lane's demand.

    Raises:
        KeyError: If the language is not in the plan.
    """
    from mixture import lanes

    config = config or Config()
    entry = next((p for p in plan() if p.code == code), None)
    if entry is None:
        raise KeyError(f"no language {code!r} in the plan")
    return entry.share_of_indic * lanes.get("indic").share * config.run_tokens


def measure_readability(sample_lines: int = 400) -> dict[str, tuple[float, float]]:
    """Recompute the table from FLORES-200, wherever it is present.

    Args:
        sample_lines: Lines per language to tokenise.

    Returns:
        Code to (`[UNK]` share, tokens per word), empty when FLORES is absent.
    """
    from datacleaning.config import FLORES_DEV
    from datacleaning.tokens import count

    if not FLORES_DEV.exists():
        return {}
    out: dict[str, tuple[float, float]] = {}
    for code in MEASURED:
        path = FLORES_DEV / f"{code}.dev"
        if not path.exists():
            continue
        text = "\n".join(path.read_text(encoding="utf-8").splitlines()[:sample_lines])
        counted = count(text)
        out[code] = (counted.unk_share, counted.fertility)
    return out


def main() -> None:
    """Print the schedule, and the measurement it rests on."""
    entries = plan()
    print(f"{'language':<24}{'script':<7}{'[UNK]':>7}{'tok/word':>9}{'wave':>9}{'share':>7}")
    for p in entries:
        share = f"{p.share_of_indic:.0%}" if p.share_of_indic else "—"
        print(f"{p.name:<24}{p.script:<7}{p.unk:>6.1%}{p.fertility:>9.2f}{p.wave:>9}{share:>7}")

    print(f"\nscheduled: {len(scheduled())}   blocked: {len(blocked())}")
    print(f"shares of the Indic lane sum to {shares_sum():.4f}")

    live = measure_readability()
    if not live:
        print("\nFLORES-200 absent; the table above is the cached measurement.")
        return
    drift = {
        c: (MEASURED[c][0], live[c][0]) for c in live if abs(MEASURED[c][0] - live[c][0]) > 0.01
    }
    print(f"\nre-measured {len(live)} from FLORES-200; drift over 1pp: {drift or 'none'}")


if __name__ == "__main__":
    main()
