import math
from functools import partial

import jax
import numpy as np
from jax import Array, numpy as jnp, random as jr


def antisym_product(a, b, triu_indices):
    r"""Computes $a \otimes b - b \otimes a$ where $\otimes$ is the outer product"""
    return (
        a[triu_indices[0]] * b[triu_indices[1]]
        - b[triu_indices[0]] * a[triu_indices[1]]
    )


vec_antisym_prod = jax.jit(jax.vmap(antisym_product, in_axes=(0, 0, None), out_axes=0))


@jax.jit
def la_poly_expansion(w: Array, hh: Array, bb: Array):
    """Computes the Levy area from the triplet $(W, H, b)$, where
    W is the Brownian motion, H is the space-time Levy area and b is
    the Levy area of the Brownian bridge."""
    assert w.ndim == hh.ndim == bb.ndim
    assert w.shape == hh.shape
    triu_indices = jnp.triu_indices(w.shape[-1], k=1)
    return bb + vec_antisym_prod(hh, w, triu_indices)


@jax.jit
def bb_chen(w1: Array, hh1: Array, bb1: Array, w2: Array, hh2: Array, bb2: Array):
    bm_dim = w1.shape[0]
    levy_dim = int(bm_dim * (bm_dim - 1) // 2)
    assert w1.shape == w2.shape == hh1.shape == hh2.shape == (bm_dim,), (
        f"Got w1.shape = {w1.shape}, w2.shape = {w2.shape}, hh1.shape = {hh1.shape},"
        f" hh2.shape = {hh2.shape}, expected {(bm_dim,)}."
    )
    assert bb1.shape == bb2.shape == (levy_dim,), (
        f"bb should have shape {(levy_dim,)}, got bb1.shape = {bb1.shape},"
        f" bb2.shape = {bb2.shape}."
    )

    triu_indices = jnp.triu_indices(bm_dim, k=1)
    sqrt_half = math.sqrt(0.5)
    w1 = sqrt_half * w1
    w2 = sqrt_half * w2
    hh1 = sqrt_half * hh1
    hh2 = sqrt_half * hh2
    bb1 = 0.5 * bb1
    bb2 = 0.5 * bb2

    hh_plus_hh = hh1 + hh2
    w_minus_w = w1 - w2
    w_out = w1 + w2

    bb_sum = bb1 + bb2
    bb_out = bb_sum + 0.5 * antisym_product(hh_plus_hh, w_minus_w, triu_indices)
    hh_out = 0.5 * hh_plus_hh + 0.25 * w_minus_w
    return w_out, hh_out, bb_out


@jax.jit
def bb_chen_consecutive(w: Array, hh: Array, bb: Array):
    assert w.shape[0] % 2 == 0, "The number of samples should be even"
    assert hh.shape[0] == bb.shape[0] == w.shape[0], (
        f"The number of samples should match, got w.shape = {w.shape},"
        f" hh.shape = {hh.shape}, bb.shape = {bb.shape}"
    )

    w1 = w[::2]
    w2 = w[1::2]
    hh1 = hh[::2]
    hh2 = hh[1::2]
    bb1 = bb[::2]
    bb2 = bb[1::2]
    vec_bb_chen = jax.vmap(bb_chen, in_axes=(0, 0, 0, 0, 0, 0))
    return vec_bb_chen(w1, hh1, bb1, w2, hh2, bb2)


@jax.jit
def la_chen(w1: Array, la1: Array, w2: Array, la2: Array):
    bm_dim = w1.shape[0]
    levy_dim = int(bm_dim * (bm_dim - 1) // 2)
    assert (
        w1.shape == w2.shape == (bm_dim,)
    ), f"Got w1.shape = {w1.shape}, w2.shape = {w2.shape}, expected {(bm_dim,)}"
    assert la1.shape == la2.shape == (levy_dim,), (
        f"la should have shape {(levy_dim,)}, got la1.shape = {la1.shape},"
        f" la2.shape = {la2.shape}."
    )
    triu_indices = jnp.triu_indices(bm_dim, k=1)

    w1 = np.sqrt(0.5) * w1
    w2 = np.sqrt(0.5) * w2
    la1 = 0.5 * la1
    la2 = 0.5 * la2

    w_out = w1 + w2
    la_out = la1 + la2 + 0.5 * antisym_product(w1, w2, triu_indices)
    return w_out, la_out


@jax.jit
def la_chen_consecutive(w: Array, la: Array):
    """Computes Chen's relation for the tuple $(W, A)$, where
    W is the Brownian motion and A is the space-space Levy area.
    The number of resulting samples is half of the number of input samples.
    The samples are paired with their next or previous sample.

    **Arguments:**
        - `w`: `Array` of shape `(2n, bm_dim)`
        - `la`: `Array` of shape `(2n, levy_dim)` where
                `levy_dim = bm_dim * (bm_dim - 1) // 2`

    **Returns:**
        - `Array` of shape `(n, bm_dim)` W
        - `Array` of shape `(n, levy_dim)` A
    """

    assert w.shape[0] % 2 == 0, "The number of samples should be even"

    w1 = w[::2]
    w2 = w[1::2]
    la1 = la[::2]
    la2 = la[1::2]
    vec_la_chen = jax.vmap(la_chen, in_axes=(0, 0, 0, 0))
    return vec_la_chen(w1, la1, w2, la2)


@partial(jax.jit, static_argnames=["dim"])
def vector_to_matrix(vector, triu_indices, dim):
    """Converts a flattened upper triangle into a skew-symmetric matrix."""
    levy_dim = (dim * (dim - 1)) // 2
    assert vector.shape == (levy_dim,)
    matrix = jnp.zeros((dim, dim), dtype=vector.dtype)
    matrix = matrix.at[triu_indices].set(vector)
    matrix = matrix.at[triu_indices[1], triu_indices[0]].set(-vector)
    return matrix


def midpoint_bridge_wh(key, w_1: Array, hh_1: Array):
    # Generating the midpoint brownian bridge for (W, H) from s = 0 to u = 1
    assert w_1.shape == hh_1.shape
    su = 1.0 - 0.0
    root_su = math.sqrt(su)
    bhh_su = su * hh_1
    z1_key, z2_key = jr.split(key, 2)
    z1 = jr.normal(z1_key, w_1.shape, w_1.dtype)
    z2 = jr.normal(z2_key, w_1.shape, w_1.dtype)
    z = z1 * (root_su / 4)
    n = z2 * jnp.sqrt(su / 12)

    w_term1 = w_1 / 2
    w_term2 = 3 / (2 * su) * bhh_su + z
    w_st = w_term1 + w_term2
    w_tu = w_term1 - w_term2
    w_st_tu = (w_st, w_tu)

    bhh_term1 = bhh_su / 8 - su / 4 * z
    bhh_term2 = su / 4 * n
    bhh_st = bhh_term1 + bhh_term2
    bhh_tu = bhh_term1 - bhh_term2
    hh_st = bhh_st / (su / 2)
    hh_tu = bhh_tu / (su / 2)
    hh_st_tu = (hh_st, hh_tu)
    return w_st_tu, hh_st_tu
