"""Optimizers and learning-rate schedules, measured on a small language model.

Five experiments live in `optimizers.experiments`: Adam reproduced by hand, bias correction switched
off, the per-layer update-to-weight ratio with and without warmup, cosine against WSD stopped early,
and a learning-rate sweep across widths in the standard parametrization and in muP. The README's
"How the pieces fit" lists every module. Importing the package needs numpy only; training needs the
`train` extra (torch).
"""
