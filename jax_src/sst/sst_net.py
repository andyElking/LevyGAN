# SST refers to space-space-time and is used to compute the integral
# $\int_0^1 W^2_t dt$ where $W$ is a one-dimensional Brownian motion.

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array

from ..generator import AbstractNet, AbstractLayer, make_layers


class SSTNet(AbstractNet):
    layers: list[AbstractLayer]
    slope: float = eqx.field(static=True)

    @property
    def dtype(self):
        return self.layers[0].dtype

    @property
    def noise_size(self):
        # The input size is 1 dim of W, 1 dim of H and the rest is noise
        return self.layers[0].in_features - 2

    def __init__(self, layers: list[AbstractLayer], slope):
        self.layers = layers
        self.slope = slope

    @jax.jit
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x, slope=self.slope)
        return jnp.abs(jnp.real(x))

    @jax.jit
    def generate_c(self, key, w: Array, hh: Array):
        bsz = w.shape[0]
        if w.ndim == 1:
            w = w[:, None]
            hh = hh[:, None]
        assert w.shape == hh.shape == (bsz, 1)
        noise = jr.normal(key, (bsz, self.noise_size), dtype=self.dtype)
        inputs = jnp.concatenate([w, hh, noise], axis=-1)
        out = self(inputs)
        assert out.shape == (bsz, 1)
        return out


def make_sst_net(
    key,
    noise_size,
    hidden_size,
    num_layers,
    slope,
    use_multlayer,
    dtype,
    use_batch_norm=False,
    use_activation=True,
) -> SSTNet:
    in_features = noise_size + 2
    layers = make_layers(
        key,
        in_features,
        hidden_size,
        num_layers,
        use_multlayer,
        dtype,
        use_batch_norm,
        use_activation,
    )
    return SSTNet(layers, slope)


def load_sst_net(
    path,
    noise_size,
    hidden_size,
    num_layers,
    slope,
    use_multlayer: bool,
    dtype,
    use_batch_norm=False,
    use_activation=True,
) -> SSTNet:
    multlayer = "_mult_layer" if use_multlayer else ""
    name = f"sst_net_nsz{noise_size}_nl{num_layers}_hd{hidden_size}{multlayer}.pckl"
    file_name = path + name
    mould = make_sst_net(
        jr.PRNGKey(0),
        noise_size,
        hidden_size,
        num_layers,
        slope,
        use_multlayer,
        dtype,
        use_batch_norm,
        use_activation,
    )
    return eqx.tree_deserialise_leaves(file_name, mould)
