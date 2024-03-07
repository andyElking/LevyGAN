from functools import partial
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from numpy.testing import assert_array_almost_equal
import equinox as eqx


@jax.jit
def matrix_to_vector(matrix, triu_indices):
    return matrix[triu_indices]


@partial(jax.jit, static_argnames=['n'])
def vector_to_matrix(vector, triu_indices, n):
    matrix = jnp.zeros((n, n), dtype=vector.dtype)
    matrix = matrix.at[triu_indices].set(vector)
    matrix = matrix.at[triu_indices[1], triu_indices[0]].set(-vector)
    return matrix


def make_layers(key, noise_size, hidden_size, num_layers):
    keys = jr.split(key, num_layers)
    in_dim = 2 * (noise_size + 1)
    layers = [eqx.nn.Linear(in_dim, hidden_size, key=keys[1])]
    for key in keys[1:-1]:
        layers.append(eqx.nn.Linear(hidden_size, hidden_size, key=key))
    layers.append(eqx.nn.Linear(hidden_size, 1, key=keys[-1]))
    return layers


class Net(eqx.Module):
    layers: list
    noise_size: int = eqx.field(static=True)

    def __init__(self, layers):
        self.layers = layers
        self.noise_size = layers[0].in_features // 2 - 1

    def __call__(self, x: Array):
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x)


def make_net(key, noise_size, hidden_size, num_layers):
    layers = make_layers(key, noise_size, hidden_size, num_layers)
    return Net(layers)


@partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, None), out_axes=0)
def bridge_flipping(w, hh, bb, rad, rad_0, triu_indices):
    n = w.shape[-1]
    len_triu = n * (n - 1) // 2
    assert triu_indices[0].shape == triu_indices[1].shape == (len_triu,)
    assert n == hh.shape[-1] == rad.shape[-1]
    assert bb.shape[-1] == len_triu
    assert w.ndim == hh.ndim == rad.ndim == bb.ndim == rad_0.ndim + 1, \
        f"{w.shape=}, {hh.shape=}, {rad.shape=}, {bb.shape=}, {rad_0.shape=}"
    rad_bb = rad[triu_indices[0]] * rad[triu_indices[1]] * bb
    rad_hh = rad * hh
    hw_outer = jnp.outer(rad_hh, w)
    hw_minus_wh = (hw_outer - hw_outer.T)[triu_indices]
    return rad_0 * (hw_minus_wh + rad_bb)


def arrange_pairnet_inputs(hh, noise, triu_indices):
    """Arrange inputs for the pairnet.

    **Arguments:**
        - `hh`: jax.Array of shape `(..., n)`
        - `noise`: jax.Array of shape `(..., n, noise_size)`
        - `triu_indices`: tuple of two jax.Array of shape `(triu_len,)`
            where `triu_len = n * (n - 1) // 2`

    **Returns:**
        `jax.Array` of shape `(..., triu_len, 2*(noise_size+1))`
        """
    # noise: (..., n, noise_size)
    n = hh.shape[-1]
    len_triu = n * (n - 1) // 2
    assert triu_indices[0].shape == triu_indices[1].shape == (len_triu,)
    assert n == noise.shape[-2]

    hh_expanded = jnp.expand_dims(hh, axis=hh.ndim)
    assert noise.ndim == hh_expanded.ndim

    hh_noise = jnp.concatenate((hh_expanded, noise), axis=-1)
    hh_noise1 = hh_noise[..., triu_indices[0], :]
    hh_noise2 = hh_noise[..., triu_indices[1], :]
    return jnp.concatenate((hh_noise1, hh_noise2), axis=-1)


@jax.jit
def generate_samples(key, net: Net, triu_indices, w: Array, hh: Optional[Array]):
    num_samples = w.shape[0]
    bm_dim = w.shape[1]
    triu_len = bm_dim * (bm_dim - 1) // 2
    noise_size = net.noise_size
    if hh is None:
        key_noise, key_rad_0, key_rad, key_hh = jr.split(key, 4)
        hh = np.sqrt(1/12) * jr.normal(key_hh, shape=(num_samples, bm_dim), dtype=w.dtype)
    else:
        key_noise, key_rad_0, key_rad = jr.split(key, 3)

    assert w.shape == hh.shape == (num_samples, bm_dim)
    assert triu_indices[0].shape == triu_indices[1].shape == (triu_len,)

    noise = jr.normal(key_noise, shape=(num_samples, bm_dim, noise_size), dtype=w.dtype)
    inputs = arrange_pairnet_inputs(hh, noise, triu_indices)  # (num_samples, triu_len, 2*(noise_size+1))
    vec_net = jax.jit(jax.vmap(jax.vmap(net, in_axes=0), in_axes=0))
    bb_unsqueezed = vec_net(inputs)  # (num_samples, triu_len, 1)
    assert bb_unsqueezed.shape == (num_samples, triu_len, 1)
    bb = jnp.squeeze(bb_unsqueezed, axis=-1)  # (num_samples, triu_len)
    rad_0 = jr.rademacher(key_rad_0, shape=(num_samples,), dtype=w.dtype)
    rad = jr.rademacher(key_rad, shape=(num_samples, bm_dim), dtype=w.dtype)
    la = bridge_flipping(w, hh, bb, rad, rad_0, triu_indices)  # Levy area
    return w, hh, la


def init_inputs(key, num_samples, bm_dim):
    key_w, key_hh = jr.split(key, 2)
    w = jr.normal(key_w, shape=(num_samples, bm_dim), dtype=jnp.float32)
    hh = jnp.sqrt(1/12) * jr.normal(key_hh, shape=(num_samples, bm_dim), dtype=jnp.float32)
    triu_indices = jnp.triu_indices(bm_dim, k=1)
    return w, hh, triu_indices