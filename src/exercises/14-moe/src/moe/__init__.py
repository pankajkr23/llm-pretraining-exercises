"""Turning a trained dense model into a mixture of experts, and showing that it keeps training.

`layer` holds the MoE feed-forward layer, built by copying a dense MLP into every expert, with a
float32 router and bias-only load balancing; `train` runs a dense or converted model to a token
budget and writes the training log; `experiments` checks the conversion changes nothing, compares
two routers and continues the MoE against the dense model. Training needs the `train` extra (torch).
"""
