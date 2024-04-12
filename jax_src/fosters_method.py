from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
import numpy as np


@jax.jit
@partial(jax.vmap, in_axes=(0, None, 0, 0, 0))
def mom4_jax(
    keys, triu_indices, w: Array, hh_in: Optional[Array], kk_in: Optional[Array]
):
    bm_dim = w.shape[-1]
    levy_dim = int(bm_dim * (bm_dim - 1) // 2)

    key_hh, key_kk, key_exp, key_ber, key_uni, key_rad = jr.split(keys, 6)
    if hh_in is None:
        hh = np.sqrt(1 / 12) * jr.normal(key_hh, (bm_dim,))
    else:
        hh = hh_in
    if kk_in is None:
        kk = np.sqrt(1 / 720) * jr.normal(key_kk, (bm_dim,))
    else:
        kk = kk_in

    squared_kk = jnp.square(kk)
    C = jr.exponential(key_exp, (bm_dim,)) * (8 / 15)
    c = np.sqrt(1 / 3) - (8 / 15)
    p = 21130 / 25621
    ber = jr.bernoulli(key_ber, p, shape=(levy_dim,))
    uni = jr.uniform(key_uni, shape=(levy_dim,), minval=-np.sqrt(3), maxval=np.sqrt(3))
    rademacher = jr.rademacher(key_rad, shape=(levy_dim,))

    ksi = ber * uni + (1 - ber) * rademacher

    C_plus_c = C + c
    sigma = (3 / 28) * (C_plus_c[triu_indices[0]] * C_plus_c[triu_indices[1]])

    sigma = sigma + (144 / 28) * (
        squared_kk[triu_indices[0]] + squared_kk[triu_indices[1]]
    )
    sigma = jnp.sqrt(sigma)

    w_i = w[triu_indices[0]]
    w_j = w[triu_indices[1]]
    hh_i = hh[triu_indices[0]]
    hh_j = hh[triu_indices[1]]
    kk_i = kk[triu_indices[0]]
    kk_j = kk[triu_indices[1]]

    tilde_a = ksi * sigma

    la_out = hh_i * w_j - w_i * hh_j + 12.0 * (kk_i * hh_j - hh_i * kk_j) + tilde_a

    return w, hh, kk, la_out
