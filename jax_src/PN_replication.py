from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax import Array
import equinox as eqx
from .generator.generator import AbstractNet, bridge_flipping, arrange_pairnet_inputs

# def vmap_wrap(f):
#     return jax.jit(jax.vmap(jax.vmap(f, in_axes=0), in_axes=0))


def make_layers(key, noise_size, hidden_size, num_layers, dtype):
    keys = jr.split(key, num_layers)
    in_dim = 2 * (noise_size + 1)
    layers = [eqx.nn.Linear(in_dim, hidden_size, key=keys[0], use_bias=False)]
    gammas = [jr.normal(key=keys[0], shape=[1, 1, hidden_size])]
    betas = [jr.normal(key=keys[0], shape=[1, 1, hidden_size])]
    for key in keys[1:-1]:
        layers.append(eqx.nn.Linear(hidden_size, hidden_size, key=key, use_bias=False))
        gammas.append(jr.normal(key=key, shape=[1, 1, hidden_size]))
        betas.append(jr.normal(key=key, shape=[1, 1, hidden_size]))
    layers.append(eqx.nn.Linear(hidden_size, 1, key=keys[-1]))
    layers = jtu.tree_map(lambda x: x.astype(dtype), layers)
    gammas = jtu.tree_map(lambda x: x.astype(dtype), gammas)
    betas = jtu.tree_map(lambda x: x.astype(dtype), betas)
    return layers, gammas, betas


def vmap_wrap(layers):
    res = []
    for layer in layers:
        res.append(jax.jit(jax.vmap(jax.vmap(layer, in_axes=0), in_axes=0)))
    return res


def torch_to_jax(jax_net, torch_state_dict):
    layers, gammas, betas = jax_net
    where_weight = lambda l: l.weight
    where_bias = lambda l: l.bias
    new_layers = []
    for i, p in enumerate(torch_state_dict):
        new_net = eqx.tree_at(
            where_weight, layers[i], jnp.array(p["net.weight"].detach().cpu().numpy())
        )
        if i < 2:
            gammas[i] = p["batch_norm.gamma"].detach().cpu().numpy()
            betas[i] = p["batch_norm.beta"].detach().cpu().numpy()
        if i == 2:
            new_net = eqx.tree_at(
                where_bias, new_net, jnp.array(p["net.bias"].detach().cpu().numpy())
            )
        new_layers.append(new_net)

    return new_layers, gammas, betas


class Net(AbstractNet):
    layers: list
    gammas: list
    betas: list
    noise_size: int = eqx.field(static=True)
    slope: float = eqx.field(static=True)

    @property
    def dtype(self):
        return self.layers[0].weight.dtype

    def __init__(self, layers, gammas, betas, slope=0):
        self.layers = layers
        self.gammas = gammas
        self.betas = betas
        self.slope = slope
        self.noise_size = layers[0].in_features // 2 - 1

    def __call__(self, x: Array) -> Array:
        for layer, gamma, beta in zip(self.layers[:-1], self.gammas, self.betas):
            x = layer(x)
            x_mean = x.mean(0, keepdims=True)
            x_var = x.var(0, keepdims=True) + 1e-05
            x = (x - x_mean) / (x_var) * gamma + beta
            x = jax.nn.leaky_relu(x, self.slope)
        return self.layers[-1](x)


@jax.jit
def generate_la(key, net: AbstractNet, triu_indices, w: Array, hh: Optional[Array]):
    num_samples = w.shape[0]
    bm_dim = w.shape[1]
    if hh is None:
        key_noise, key_rad_0, key_rad, key_hh = jr.split(key, 4)
        hh = np.sqrt(1 / 12) * jr.normal(
            key_hh, shape=(num_samples, bm_dim), dtype=w.dtype
        )
    else:
        key_noise, key_rad_0, key_rad = jr.split(key, 3)

    assert w.shape == hh.shape == (num_samples, bm_dim)

    bb = generate_bb(key_noise, net, triu_indices, hh)

    rad_0 = jr.rademacher(key_rad_0, shape=(num_samples,), dtype=w.dtype)
    rad = jr.rademacher(key_rad, shape=(num_samples, bm_dim), dtype=w.dtype)
    la = bridge_flipping(w, hh, bb, rad, rad_0, triu_indices)  # Levy area

    return w, hh, la


@jax.jit
def generate_bb(key, net, triu_indices, hh: Array):
    num_samples = hh.shape[0]
    bm_dim = hh.shape[1]
    triu_len = bm_dim * (bm_dim - 1) // 2
    noise_size = net.noise_size
    assert triu_indices[0].shape == triu_indices[1].shape == (triu_len,)

    noise = jr.normal(key, shape=(num_samples, bm_dim, noise_size), dtype=hh.dtype)
    inputs = arrange_pairnet_inputs(
        hh, noise, triu_indices
    )  # (num_samples, triu_len, 2*(noise_size+1))

    bb_unsqueezed = net(inputs)  # (num_samples, triu_len, 1)
    assert bb_unsqueezed.shape == (num_samples, triu_len, 1)
    bb = jnp.squeeze(bb_unsqueezed, axis=-1)  # (num_samples, triu_len)
    return bb
