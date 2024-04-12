from functools import partial
from typing import Optional, Literal

import jax
from jax import Array, random as jr, numpy as jnp

from jax_src.fosters_method import mom4_jax
from jax_src.generator import AbstractNet, generate_la
from jax_src.aux_functions import la_chen_consecutive


@partial(jax.jit, static_argnames=["m", "n", "bm_dim", "use_chen"])
def _gen_mc_helper(
    m: int,
    n: int,
    bm_dim: int,
    use_chen: bool,
    dt: Array,
    triu_indices: tuple[Array, Array],
    net: Optional[AbstractNet],
    key,
):
    """A jitted helper for generate_mc_samples_jax"""

    key_w, key_la = jr.split(key)
    w = jr.normal(key_w, (m * n, bm_dim), dtype=jnp.float32)
    if net is not None:
        _, _, la = generate_la(key_la, net, triu_indices, w, None)
    else:
        keys_la = jr.split(key_la, m * n)
        _, _, _, la = mom4_jax(keys_la, triu_indices, w, None, None)
    w = jnp.sqrt(dt) * w
    la = dt * la

    out = jnp.concatenate((w, la), axis=1)
    concat_dim = bm_dim * (bm_dim + 1) // 2
    out = jnp.reshape(out, (m, n, concat_dim))
    if use_chen:
        w_chen_comb, la_chen_comb = la_chen_consecutive(w, la)
        out_chen = jnp.concatenate((w_chen_comb, la_chen_comb), axis=1)
        out_chen = jnp.reshape(out_chen, (m, n // 2, concat_dim))
        return out, out_chen
    else:
        return out


class LASamplerForMC:
    bm_dim: int
    net: Optional[AbstractNet]
    key: Array
    method: Literal["GAN", "Foster"]
    triu_indices: tuple[Array, Array]

    def __init__(self, bm_dim, net, key, method: Literal["GAN", "Foster"]):
        self.bm_dim = bm_dim
        self.key = key
        self.method = method
        if method == "GAN":
            assert (
                net is not None
            ), "If the method is 'GAN' a neural net should be provided"
            self.net = net
        else:
            self.net = None
        self.triu_indices = jnp.triu_indices(bm_dim, 1)

    def generate_mc_samples(self, m: int, n: int, dt: float, use_chen: bool = False):
        """generates samples with shape M x N x (bm_dim + levy_dim) over the time interval dt

        **Arguments:**
            M (int): number of MC paths
            N (int): number of timesteps
            dt (float): timestep
        """
        dt_jax = jnp.asarray(dt, dtype=jnp.float32)
        # Split the key, so you generate different samples each time
        self.key, gen_key = jr.split(self.key, 2)
        if use_chen:
            assert n % 2 == 0, "n should be even if use_chen is True"
        return _gen_mc_helper(
            m, n, self.bm_dim, use_chen, dt_jax, self.triu_indices, self.net, gen_key
        )
