"""Modern attention variants, in the order they were launched.

Exercise 08 covers roughly twenty ways of computing attention and of telling a model where a token
sits. The assignment is not to describe them — it is to put them in **chronological order by the
date each one actually appeared**, and to explain each as an answer to a problem that existed at
that moment.

That framing is what makes the exercise unusual, and it decides the shape of this package:

- **The evidence is a chronology, not a training run.** Nothing here trains a model. The claims are
  a set of dates, each read from the primary source, plus closed-form arithmetic for the costs the
  dates are a response to. So there is no torch, and CI verifies the whole exercise rather than
  skipping its heaviest part.
- **A date with no source is not publishable.** We were warned that an agent asked for a launch
  date will supply a confident one it has half remembered, and that is the failure mode this
  package is built to make impossible. `sources.Source` will not construct a `verified` citation
  without a URL and the verbatim string the date was read from, and `catalogue.unverified` reports
  any entry a reader could not check.
- **A mechanism with only advantages has not been understood.** `catalogue.Mechanism` refuses to
  construct without a stated trade-off, because the assignment says so and because it is true.

Modules:
    `config`     the yardstick model every cost is computed against, from the source material itself
    `cache`      the two bills: T^2 scores and a KV cache linear in T
    `sources`    the citation model — what was read, from where, quoted, and when
    `catalogue`  the mechanisms, their trade-offs, and the coverage list the assignment mandates
    `timeline`   ordering, and the eras the order reveals
"""
