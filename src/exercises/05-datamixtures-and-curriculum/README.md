# 05 · Data mixtures and curriculum

The V5 pre-training recipe: how much of each kind of data the model sees, in what order, and what
happens to every share when you check it against the data that actually exists.

**The specification is [`SPEC.md`](SPEC.md).** This file says how it is built and how to rerun it.

## The one rule this exercise runs on

Every lane's supply is **summed from the datasets named in the Session 5 inventory**, never quoted
from a slot headline. That single choice is what makes the spec defensible, and it changed an
answer on the first run: the STEM lane's itemised supply is **146B**, where the session's own
supply-check widget prices it at **250B**. Against a 240B demand those are different worlds — one
fits inside a single pass, the other needs repetition. No dataset in the inventory carries the
missing 104B, so the spec uses the itemised figure and shows its work.

## Run it

```bash
uv run python -m mixture.inventory     # lane supplies, itemised vs the session's two headlines
uv run pytest src/exercises/05-datamixtures-and-curriculum
```

## Layout

```text
src/mixture/
  config.py       every threshold in one frozen dataclass, with a fingerprint
  inventory.py    the Session 5 dataset inventory as 30 typed rows; lane supplies summed from them
tests/            every invariant, each paired with a twin that fails
PROGRESS.md       the running log: findings, decisions, and what would overturn each
```

More modules land as the spec is built; `PROGRESS.md` is the current state of play.
