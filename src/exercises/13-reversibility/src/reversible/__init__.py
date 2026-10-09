"""Reversible training: activation memory traded for compute, and what that buys in batch size.

`stack` holds the reversible rules (midpoint, blend, leapfrog) and the autograd function that
rebuilds each layer's input during the backward pass instead of storing it; `model` chains exercise
11's blocks by a chosen rule; `memory` measures what a step keeps and finds the largest batch that
fits; `train` and `experiments` run the comparisons the exercise asks for. Training needs the
`train` extra (torch).
"""
