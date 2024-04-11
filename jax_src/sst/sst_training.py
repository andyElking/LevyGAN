from math import sqrt
from types import NoneType

import equinox as eqx
import jax

from jax import Array, random as jr, numpy as jnp, tree_util as jtu

from jax_src.aux_functions import midpoint_bridge_wh
from jax_src.discriminator import marginal_wass2_error
from jax_src.sst import SSTNet


@jax.jit
def sst_chen(w1: Array, hh1: Array, c1: Array, w2: Array, hh2: Array, c2: Array):
    """Computes the Chen relation for (W, H, C)"""
    bsz = w1.shape[0]
    assert w1.shape == w2.shape == hh1.shape == hh2.shape == c1.shape == c2.shape
    assert w1.shape == (bsz,) or w1.shape == (bsz, 1)

    # Brownian scaling
    c1 = 0.25 * c1
    c2 = 0.25 * c2
    sqrt_half = sqrt(0.5)
    w1 = sqrt_half * w1
    w2 = sqrt_half * w2
    hh1 = sqrt_half * hh1
    hh2 = sqrt_half * hh2

    w = w1 + w2
    hh = 0.5 * (hh1 + hh2) + 0.25 * (w1 - w2)
    c = c1 + c2 + w1 * (0.5 * w + hh2)
    return w, hh, c


@jax.jit
def sst_chen_consecutive(w: Array, hh: Array, c: Array):
    assert w.shape[0] % 2 == 0
    w1 = w[::2]
    w2 = w[1::2]
    hh1 = hh[::2]
    hh2 = hh[1::2]
    c1 = c[::2]
    c2 = c[1::2]
    w_out, hh_out, c_out = sst_chen(w1, hh1, c1, w2, hh2, c2)
    return w_out, hh_out, c_out


def sst_loss_fixed_wh(model: tuple[SSTNet, NoneType], key) -> Array:
    """Computes the loss of the generator and the discriminator
    using Chen's relation for (W, H)."""
    bsz = 2**14
    net, _ = model
    key_hh_1, key_w_1, key_midpoint, key_model = jr.split(key, 4)
    bm_mult = 1.0
    hh_mult = bm_mult * sqrt(1 / 12)
    root2 = sqrt(2.0)

    w_1 = bm_mult * jr.normal(key_w_1, (), dtype=net.dtype)
    hh_1 = hh_mult * jr.normal(key_hh_1, (), dtype=net.dtype)
    w_1 = jnp.broadcast_to(w_1, (bsz, 1))
    hh_1 = jnp.broadcast_to(hh_1, (bsz, 1))
    bridge = midpoint_bridge_wh(key_midpoint, w_1, hh_1)

    # scale the bridge values by sqrt(2)
    (w_0_half, w_half_1), (hh_0_half, hh_half_1) = jtu.tree_map(
        lambda x: root2 * x, bridge
    )

    key_model1, key_model2, key_model3 = jr.split(key_model, 3)
    c_0_half = net.generate_c(key_model1, w_0_half, hh_0_half)
    c_half_1 = net.generate_c(key_model2, w_half_1, hh_half_1)
    w_true, hh_true, c_true = sst_chen(
        w_0_half, hh_0_half, c_0_half, w_half_1, hh_half_1, c_half_1
    )
    w_true = eqx.internal.error_if(
        w_true,
        jnp.logical_not(jnp.allclose(w_true, w_1, atol=1e-5)),
        "chen combine should be the inverse of brownian bridge, W doesnt match",
    )
    hh_true = eqx.internal.error_if(
        hh_true,
        jnp.logical_not(jnp.allclose(hh_true, hh_1, atol=1e-5)),
        "chen combine should be the inverse of brownian bridge, H doesnt match",
    )

    # Stop the gradient of the true samples
    w_true = jax.lax.stop_gradient(w_true)
    hh_true = jax.lax.stop_gradient(hh_true)
    c_true = jax.lax.stop_gradient(c_true)

    c_fake = net.generate_c(key_model3, w_true, hh_true)

    # Compute the Wasserstein distance between c_true and c_fake
    return marginal_wass2_error(c_true, c_fake)
