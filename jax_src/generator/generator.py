from abc import abstractmethod
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from equinox import AbstractVar
from jax import Array
import equinox as eqx

from ..aux_functions import antisym_product


def _squareplus(x, slope):
    return 0.5 * ((1 + slope) * x + (1 - slope) * jnp.sqrt(x**2 + 0.1))


class AbstractLayer(eqx.Module):
    dtype: AbstractVar[type]
    in_features: AbstractVar[int]
    out_features: AbstractVar[int]

    @abstractmethod
    def __call__(self, x: Array, **kwargs) -> Array:
        raise NotImplementedError


class Layer(AbstractLayer):
    weight: Array
    bias: Array
    batch_norm: bool = eqx.field(static=True)
    use_activation: bool = eqx.field(static=True)

    @property
    def dtype(self):
        return self.weight.dtype

    @property
    def in_features(self):
        return self.weight.shape[1]

    @property
    def out_features(self):
        return self.weight.shape[0]

    def __init__(
        self, weight: Array, bias: Array, batch_norm: bool, use_activation: bool
    ):
        assert weight.ndim == 2
        assert bias.shape == (weight.shape[0],)
        self.weight = weight
        self.bias = bias
        self.batch_norm = batch_norm
        self.use_activation = use_activation

    def __call__(self, x: Array, slope: float = 0.01) -> Array:
        """Applies the layer to the input.

        **Arguments:**
            - `x`: `jax.Array` of shape `(num_samples, triu_len, in_features)`
            where triu_len = bm_dim * (bm_dim - 1) // 2
            or of shape `(num_samples, in_features)`.

        **Returns:**
            `jax.Array` of shape `(num_samples, triu_len, out_features)`
            or of shape `(num_samples, out_features)` if `x` has shape
            `(num_samples, in_features)`.
        """
        assert x.ndim in (2, 3)
        assert x.shape[-1] == self.in_features
        x = jnp.tensordot(x, self.weight, axes=(-1, 1))
        if self.batch_norm:
            x = x - jnp.mean(x, axis=0, keepdims=True)
            x = x / (jnp.std(x, axis=0, keepdims=True) + 1e-6)
        dims_to_expand = (0, 1) if x.ndim == 3 else (0,)
        x = x + jnp.expand_dims(self.bias, axis=dims_to_expand)
        if self.use_activation:
            x = _squareplus(x, slope)
        return x


class MultLayer(AbstractLayer):
    weight0: Array
    weight1: Array
    weight2: Array
    bias: Array
    use_activation: bool = eqx.field(static=True)

    @property
    def dtype(self):
        return self.weight0.dtype

    @property
    def in_features(self):
        return self.weight0.shape[1]

    @property
    def out_features(self):
        return self.weight0.shape[0] + self.weight1.shape[0]

    def __init__(self, weight0, weight1, weight2, bias, use_activation: bool):
        assert weight0.ndim == weight1.ndim == weight2.ndim == 2
        assert bias.shape == (weight0.shape[0] + weight1.shape[0],)
        assert weight1.shape[1] == weight2.shape[1] == weight0.shape[1]

        self.weight0 = weight0
        self.weight1 = weight1
        self.weight2 = weight2
        self.bias = bias
        self.use_activation = use_activation

    def __call__(self, x, slope: float = 0.01):
        # x: (num_samples, triu_len, in_features) or (num_samples, in_features)
        assert x.ndim in (2, 3)
        assert x.shape[-1] == self.in_features
        x0 = jnp.tensordot(x, self.weight0, axes=(-1, 1))
        x1 = jnp.tensordot(x, self.weight1, axes=(-1, 1))
        x2 = jnp.tensordot(x, self.weight2, axes=(-1, 1))

        x12 = x1 * x2
        x012 = jnp.concatenate((x0, x12), axis=-1)
        dims_to_expand = (0, 1) if x.ndim == 3 else (0,)
        x012 = x012 + jnp.expand_dims(self.bias, axis=dims_to_expand)
        if self.use_activation:
            x012 = _squareplus(x012, slope)
        return x012


class AbstractNet(eqx.Module):
    layers: AbstractVar[list[AbstractLayer]]
    noise_size: AbstractVar[int]
    dtype: AbstractVar[type]

    @abstractmethod
    def __call__(self, x: Array) -> Array:
        raise NotImplementedError


class Net(AbstractNet):
    layers: list[AbstractLayer]
    slope: float = eqx.field(static=True)

    @property
    def dtype(self):
        return self.layers[0].dtype

    @property
    def noise_size(self):
        # The input size is 1 dim of W, 1 dim of H and the rest is noise
        return self.layers[0].in_features // 2 - 1

    def __init__(self, layers, slope):
        self.layers = layers
        self.slope = slope

    def __call__(self, x: Array) -> Array:
        for layer in self.layers:
            x = layer(x, slope=self.slope)
        return jnp.real(x)


def bridge_flipping(w, hh, bb, rad, rad_0, triu_indices):
    n = w.shape[-1]
    len_triu = n * (n - 1) // 2
    assert triu_indices[0].shape == triu_indices[1].shape == (len_triu,)
    assert n == hh.shape[-1] == rad.shape[-1]
    assert bb.shape[-1] == len_triu
    assert rad_0.shape[-1] == 1
    assert (
        w.shape[:-1]
        == hh.shape[:-1]
        == rad.shape[:-1]
        == rad_0.shape[:-1]
        == bb.shape[:-1]
    ), f"{w.shape=}, {hh.shape=}, {rad.shape=}, {bb.shape=}, {rad_0.shape=}"
    rad_bb = rad[..., triu_indices[0]] * rad[..., triu_indices[1]] * bb
    rad_hh = rad * hh
    vec_antisym_prod = jax.vmap(antisym_product, in_axes=(0, 0, None), out_axes=0)
    hw_minus_wh = vec_antisym_prod(rad_hh, w, triu_indices)
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
    assert bb.shape == (num_samples, bm_dim * (bm_dim - 1) // 2)

    rad_0 = jr.rademacher(key_rad_0, shape=(num_samples, 1), dtype=w.dtype)
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
    assert inputs.shape == (num_samples, triu_len, 2 * (noise_size + 1))
    bb_unsqueezed = net(inputs)  # (num_samples, triu_len, 1)
    assert bb_unsqueezed.shape == (num_samples, triu_len, 1)
    bb = jnp.squeeze(bb_unsqueezed, axis=-1)  # (num_samples, triu_len)
    return bb


def init_inputs(key, num_samples, bm_dim, dtype):
    key_w, key_hh = jr.split(key, 2)
    w = jr.normal(key_w, shape=(num_samples, bm_dim), dtype=dtype)
    hh = jnp.sqrt(1 / 12) * jr.normal(key_hh, shape=(num_samples, bm_dim), dtype=dtype)
    triu_indices = jnp.triu_indices(bm_dim, k=1)
    return w, hh, triu_indices
