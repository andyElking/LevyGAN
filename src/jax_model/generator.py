from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
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

key = jr.PRNGKey(0)


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

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, input):
        x = input
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return input, self.layers[-1](x)


@partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, None))
def bridge_flipping(w, hh, bb, rad, rad_0, triu_indices):
    n = w.shape[-1]
    len_triu = n * (n - 1) // 2
    assert triu_indices[0].shape == triu_indices[1].shape == (len_triu,)
    assert n == hh.shape[-1] == rad.shape[-1]
    assert bb.shape[-1] == len_triu
    assert w.ndim == hh.ndim == rad.ndim == bb.ndim == rad_0.ndim + 1
    rad_bb = rad[triu_indices[0]] * rad[triu_indices[1]] * bb
    rad_hh = rad * hh
    hw_outer = jnp.outer(rad_hh, w)
    hw_minus_wh = (hw_outer - hw_outer.T)[triu_indices]
    return rad_0 * (hw_minus_wh + rad_bb)


def arrange_pairnet_inputs(hh, noise, triu_indices):
    # noise: (..., n, noise_size)
    n = hh.shape[-1]
    len_triu = n * (n - 1) // 2
    assert triu_indices[0].shape == triu_indices[1].shape == (len_triu,)
    assert n == noise.shape[-2]
    assert noise.ndim == hh.ndim
    hh_expanded = jnp.expand_dims(hh, axis=hh.ndim)
    hh_noise = jnp.concatenate((hh_expanded, noise), axis=-1)
    hh_noise1 = hh_noise[..., triu_indices[0], :]
    hh_noise2 = hh_noise[..., triu_indices[1], :]
    return jnp.concatenate((hh_noise1, hh_noise2), axis=-1)