# Re-run, 2026-10-24: the full experiment on the code that fixed O8

**Why.** `PROGRESS.md` O8 found three defects while running the notebook's LITE preset on Apple's
GPU. Evaluation built an autograd graph nobody used. The largest-batch search left its models and
cached blocks to the run that followed it. And a search that measured 0 was read as "not measured".
Fixing them changed the package, so its `code_digest` no longer matches the one the published
bundles in [`../../`](../../) carry. A fix that moves the code this exercise publishes from cannot
land without the run that shows what it does to the numbers. `test_a_code_change_comes_with_a_rerun`
fails until a re-run exists.

**The historical bundles are unchanged.** `RESULTS.md`, the page and exercise 14 all still use them.
Exercise 14 converted the published `baseline-full.pt`, so this run wrote its own checkpoint to
`artifacts/checkpoints/reruns/2026-10-24/` (gitignored) and left that one in place.

**How.** `tools/run_experiments.py` with its defaults (FULL, both stages, the second in its own
process), on commit `05e8cfe`, which is the `git_sha` every bundle here records. The run started at
13:58 IST and finished at 15:55, wrapped in `caffeinate -i`. [`RESULTS.md`](RESULTS.md) and
`web-data.js` were rendered from these bundles by `tools/render_results.py`'s own `render` and
`render_page_data`.

Compared with the historical run:

- **Same:** settings (identical `config_fingerprint` on all four bundles), corpus and tokenizer
  digests, torch 2.13.0, device (`mps`), and the 8 GiB cap.
- **Different:** the code (`sha256:882b596d…` against `sha256:d8ec4764…`), and the operating system
  (macOS 27.0.1 against 26.6.2).
- **Not recorded:** this run had no vitals watchdog, and the sandbox it was launched from cannot
  read the process list. So whatever else the machine was doing is unknown. A web browser was using
  about 2.7 CPU cores when the run began.

| figure | historical | this re-run |
| --- | ---: | ---: |
| trial chosen | leapfrog, h = 0.25 | leapfrog, h = 0.25 |
| its trial validation loss | 5.0171 | 5.0166 |
| every trial loss, largest difference | — | 0.0156 (blend, h = 1.0) |
| bytes kept for backward, baseline ÷ reversible | **58.7×** | **58.7×** |
| final loss at batch 32, baseline | 3.0526 | 3.0503 |
| final loss at batch 32, reversible | 3.1020 | 3.0989 |
| measured largest batch, baseline | 81 | 81 |
| measured largest batch, reversible | 496 | 467 |
| reversible ÷ baseline, largest batch | **6.1×** | **5.8×** |
| run near the largest batch: batch, steps | 421, 463 | 396, 493 |
| its final loss | 3.9370 | 3.8912 |
| baseline trial speeds (tokens/s) | 34,443–42,324 | 25,707–40,355 |
| reversible trial speeds (tokens/s) | 25,556–27,536 | 25,364–27,072 |
| speed verdict from the trials | every reversible slower than every baseline | **overlap: not established** |
| long pair at batch 32 (tokens/s) | 28,728 against 25,807 | 30,933 against 26,119 |
| baseline drift, trial → long run | 1.44× | 1.02× |

Sources: [`RESULTS.md`](RESULTS.md) here against [`../../../RESULTS.md`](../../../RESULTS.md). The
bundles are [`trials.json`](trials.json), [`fixed_batch.json`](fixed_batch.json),
[`max_batch.json`](max_batch.json) and [`max_batch_run.json`](max_batch_run.json).

## Conclusion

**Memory: holds.** The bytes kept for backward are a count, not a timing, and they are identical:
58.7× fewer for the reversible stack. The measured largest batch moved from 496 to 467, and the
ratio from 6.1× to 5.8×. This is the first time the search has been run twice, so 6% is the only
spread on record between two searches. The headline is "about six times", and no finer.

**Loss: holds.** The trials chose the same rule and step size. Every trial loss is within 0.016 of
its historical value, and both 50-million-token losses are within 0.004. The gap between them
(+0.0486 against +0.0494) is the same. The run near the largest batch used a smaller batch, because
the search found less, and took more steps. Its loss is 0.046 lower, which is consistent with the
published explanation (fewer, larger steps train less far) without testing it.

**Speed: diverges.** The fix touched nothing inside the timed steps (evaluation is off the clock),
but the published verdict does not survive this run:

- **The reversible model's speed is stable.** In both runs its nine trials sit between about 25,400
  and 27,500 tokens per second.
- **The baseline's is not.** Its three trials differ only in learning rate, which costs no compute,
  yet they ran at 42k, 41k and 34k in the published run and 40k, 30k and 26k in this one. Each run's
  trials got slower in the order they ran. That fits a machine slowing down as it warms up, or
  background load. Neither was measured.
- **The trial verdict flips.** The published trials put every reversible candidate below every
  baseline run. In this run the slowest baseline trial (25,707) sits inside the reversible range, so
  the renderer's verdict is "mixed".
- **The long pair can be sized this time.** The baseline drifted only 1.02× between its trial and
  its long run, and the long pair differs by 1.18×. So by the renderer's own rule, this pair sizes
  the reversible model as 16% slower. The historical long pair could not be sized, because its drift
  was 1.44×.

**The honest reading across both runs:** the reversible model was never measured faster. One long
pair puts it about 16% slower. But the
trial comparison the published verdict rested on is noisier than the effect: the baseline's own
speed varies by 1.57× across three runs that cost the same. **A speed claim needs a noise floor
taken under a measured machine state** — the next hypothesis below.

**The next hypothesis.** Run the three baseline trials and the chosen reversible trial interleaved,
several times each, with the vitals watchdog recording the thermal state. Then rank speed only if
the reversible range sits outside the baseline's. If the ordering of the baseline trials still sets
their speed, the trial comparison is measuring the machine rather than the model.
