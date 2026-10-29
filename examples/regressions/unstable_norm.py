"""Injected implementation-regression fixture. Never an optimization candidate.

Dropping maximum scaling looks simpler but overflows on large finite rows.
Used to test independent rejection, not attribution
of a historical regression.
"""
def unstable(x):
    import mlx.core as mx
    return x/mx.maximum(mx.linalg.norm(x,axis=1,keepdims=True),1e-12)
