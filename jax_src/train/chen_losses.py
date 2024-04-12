from math import sqrt

import jax
from jax import Array, numpy as jnp, random as jr, tree_util as jtu, lax
import equinox as eqx

from ..discriminator import AbstractDiscriminator
from ..generator import AbstractNet, generate_bb, generate_la
from ..aux_functions import (
    bb_chen_consecutive,
    la_chen_consecutive,
    bb_chen,
    midpoint_bridge_wh,
)


def bb_loss(model: tuple[AbstractNet, AbstractDiscriminator], key) -> Array:
    """Computes the loss of the generator and the discriminator
    using Chen's relation for (H, b). Does not use Bridge Flipping."""
    bsz = 2**12
    net, discriminator = model
    bm_dim = discriminator.bm_dim
    triu_indices = jnp.triu_indices(bm_dim, k=1)
    key_hh, key_w, key_model = jr.split(key, 3)
    hh = sqrt(1 / 12) * jr.normal(key_hh, (bsz, bm_dim), dtype=net.dtype)

    w = jr.normal(key_w, (bsz, bm_dim), dtype=net.dtype)  # needed only for Chen

    bb_fake = generate_bb(key_model, net, triu_indices, hh)
    samples_fake = (hh, bb_fake)

    _, hh_true, bb_true = bb_chen_consecutive(w, hh, bb_fake)
    samples_true = (hh_true, bb_true)
    # Stop the gradient of the true samples
    samples_true = jtu.tree_map(lax.stop_gradient, samples_true)

    return jnp.mean(discriminator(samples_true, samples_fake))


def la_loss(model: tuple[AbstractNet, AbstractDiscriminator], key) -> Array:
    """Computes the loss of the generator and the discriminator
    using Chen's relation for (W, A). Uses Bridge Flipping."""
    bsz = 2**12
    net, discriminator = model
    bm_dim = discriminator.bm_dim
    triu_indices = jnp.triu_indices(bm_dim, k=1)
    key_w, key_model = jr.split(key, 2)
    w = jr.normal(key_w, (bsz, bm_dim), dtype=net.dtype)

    _, _, la_fake = generate_la(key_model, net, triu_indices, w, None)
    samples_fake = (w, la_fake)

    samples_true = la_chen_consecutive(w, la_fake)
    # Stop the gradient of the true samples
    samples_true = jtu.tree_map(lax.stop_gradient, samples_true)

    return jnp.mean(discriminator(samples_true, samples_fake))


def bb_loss_fixed_wh(model: tuple[AbstractNet, AbstractDiscriminator], key) -> Array:
    """The same as `bb_loss` but with fixed w and h."""
    bsz = 2**14
    net, discriminator = model
    bm_dim = discriminator.bm_dim
    triu_indices = jnp.triu_indices(bm_dim, k=1)
    key_hh_1, key_w_1, key_midpoint, key_model = jr.split(key, 4)
    bm_mult = 1.0
    hh_mult = bm_mult * sqrt(1 / 12)
    root2 = sqrt(2.0)

    w_1 = bm_mult * jr.normal(key_w_1, (bm_dim,), dtype=net.dtype)
    hh_1 = hh_mult * jr.normal(key_hh_1, (bm_dim,), dtype=net.dtype)
    w_1 = jnp.broadcast_to(w_1, (bsz, bm_dim))
    hh_1 = jnp.broadcast_to(hh_1, (bsz, bm_dim))
    bridge = midpoint_bridge_wh(key_midpoint, w_1, hh_1)

    # scale the bridge values by sqrt(2)
    (w_0_half, w_half_1), (hh_0_half, hh_half_1) = jtu.tree_map(
        lambda x: root2 * x, bridge
    )

    key_model1, key_model2, key_model3 = jr.split(key_model, 3)

    bb_0_half = generate_bb(key_model1, net, triu_indices, hh_0_half)
    bb_half_1 = generate_bb(key_model2, net, triu_indices, hh_half_1)

    fixed_bb_chen = jax.vmap(bb_chen, in_axes=(0, 0, 0, 0, 0, 0), out_axes=(0, 0, 0))

    w_true, hh_true, bb_true = fixed_bb_chen(
        w_0_half, hh_0_half, bb_0_half, w_half_1, hh_half_1, bb_half_1
    )

    hh_true = eqx.internal.error_if(
        hh_true,
        jnp.logical_not(
            jnp.logical_and(
                jnp.allclose(hh_true, hh_1, atol=1e-5),
                jnp.allclose(w_true, w_1, atol=1e-5),
            )
        ),
        "chen combine should be the inverse of brownian bridge, H doesnt match",
    )

    # Stop the gradient
    hh_true = lax.stop_gradient(hh_true)
    bb_true = lax.stop_gradient(bb_true)

    samples_true = (hh_true, bb_true)

    bb_fake = generate_bb(key_model3, net, triu_indices, hh_true)
    samples_fake = (hh_true, bb_fake)

    return jnp.mean(discriminator(samples_true, samples_fake))
