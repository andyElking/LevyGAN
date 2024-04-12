from functools import partial

import equinox as eqx
import jax
from jax import Array, numpy as jnp, random as jr
from jax._src.scipy.linalg import expm

from .discriminator import AbstractDiscriminator


class AntiHermitian(eqx.Module):
    data: Array

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self):
        return self.data.ndim

    def get_matrix(self) -> Array:
        """Returns an anti-hermitian matrix."""
        conj_transpose = jnp.conj(jnp.swapaxes(self.data, -1, -2))
        return (self.data - conj_transpose) / 2.0


def init_transform(key, bm_dim, m, n, dtype):
    """Initializes the transformation tensor.
    **Arguments:**
        - `key`: `jax.random.PRNGKey`
        - `bm_dim`: dimension of the Brownian motion
        - `m`: number of unitary transformations over which we average
        - `n`: the dimension of the Lie algebra

    **Returns:**
        `AntiHermitian` of shape `(m, d, n, n)` where `d = bm_dim + (bm_dim * (bm_dim - 1) // 2)`
        since that dimension will be multiplied against the concat of H and bb.
    """

    d = bm_dim + (bm_dim * (bm_dim - 1) // 2)

    tensor_imag = jr.normal(key, (m, d, n, n), dtype=dtype)
    tensor_real = jr.normal(key, (m, d, n, n), dtype=dtype)
    tensor = 1 * (tensor_real + 1j * tensor_imag)

    return AntiHermitian(tensor)


@partial(jax.vmap, in_axes=(0, None), out_axes=0)
def _u_m(x: Array, mm: Array):
    """
    Computes $\mathcal{U}_m(x) = \exp(M(x))$, where $\exp$ is the matrix exponential.

    **Arguments:**
        - `x`: `Array` of shape `(num_samples, d)` (since we are using `vmap`
                                                    over the first axis of x)
        - `mm`: `Array` of shape `(d, n, n)`

    **Returns:**
        `Array` of shape `(num_samples, n, n)` (due to `vmap`)
    """
    assert x.ndim == 1
    assert x.shape[0] == mm.shape[-3], (
        f"The shape of x and m do not match, should be (d,) and "
        f"(d, n, n) but got {x.shape=} and {mm.shape=} respectively."
    )
    xm = jnp.tensordot(x, mm, axes=(0, mm.ndim - 3))
    return expm(xm)


def _eucf(x, mm):
    """Computes the empirical Unitary Characteristic Function
    (eUCF) of the samples `x`"""
    assert x.ndim == 2
    assert mm.ndim == 3
    # Average over the number of samples
    # u_m returns a (num_samples, n, n) tensor where n is the dimension
    # of the Lie algebra and num_samples is the number of samples in x
    return jnp.mean(_u_m(x, mm), axis=0)


def _sq_eucf_dist(x: Array, y: Array, mm: Array):
    """Computes the square empirical Unitary Characteristic Function (UCF) distance
    between the samples `x` and `y` using a single transformation tensor `m`."""
    assert x.ndim == y.ndim == 2
    diff = _eucf(x, mm) - _eucf(y, mm)
    n = mm.shape[-1]
    assert diff.shape == (n, n)
    sq_hs_dist = jnp.trace(diff @ jnp.conj(diff.T))
    return sq_hs_dist


class UCFDiscriminator(AbstractDiscriminator):
    M: AntiHermitian
    bm_dim: int = eqx.field(static=True)

    def __init__(self, M, bm_dim):
        self.M = M
        self.bm_dim = bm_dim

    @property
    def d(self):
        """The dimension of the inputspace."""
        return self.bm_dim + (self.bm_dim * (self.bm_dim - 1) // 2)

    @property
    def m(self):
        """The number of transformations over which we average."""
        return self.M.data.shape[0]

    @property
    def n(self):
        """The dimension of the Lie algebra."""
        return self.M.data.shape[-1]

    def __call__(
        self, samples_true: tuple[Array, Array], samples_fake: tuple[Array, Array]
    ) -> Array:
        # Either (x = hh, y = bb), or (x = w, y = la)
        x_true, y_true = samples_true
        x_fake, y_fake = samples_fake
        assert x_true.ndim == x_fake.ndim == 2
        assert y_true.ndim == y_fake.ndim == 2
        assert x_true.shape[1] == x_fake.shape[1] == self.bm_dim
        assert (
            y_true.shape[1] == y_fake.shape[1] == self.bm_dim * (self.bm_dim - 1) // 2
        )

        x_y_fake = jnp.concatenate((x_fake, y_fake), axis=1)
        x_y_true = jnp.concatenate((x_true, y_true), axis=1)

        # vectorize the squared eucf distance to be used for all the Ms
        vec_eucf_dist = jax.vmap(_sq_eucf_dist, in_axes=(None, None, 0), out_axes=0)
        mm: Array = self.M.get_matrix()
        # Average over the number of transformations and take the square root
        avg = jnp.mean(vec_eucf_dist(x_y_true, x_y_fake, mm))
        return jnp.sqrt(avg)
