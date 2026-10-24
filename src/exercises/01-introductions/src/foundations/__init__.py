"""Four foundational claims about neural networks, each as a small seeded experiment.

`data` builds the inputs (concentric rings, separable blobs, interleaved spirals, noisy half-moons
and a toy three-category grammar); `models` holds the networks; `train` is the one training loop and
the device picker; `experiments` runs the four claims and returns plain results. Every setting
lives in `config`. Training needs the `train` extra (torch); importing this package alone does not.
"""
