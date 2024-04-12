import jax
from jax import Array
import jax.numpy as jnp
from ott.geometry.pointcloud import PointCloud
from ott.solvers.linear.univariate import UnivariateSolver
from ott.problems.linear.linear_problem import LinearProblem
import equinox as eqx
import equinox.internal as eqxi

from .discriminator import AbstractDiscriminator


@jax.jit
def marginal_wass2_error(x: Array, y: Array) -> Array:
    """Computes the marginal Wasserstein-2 distances between two distributions."""
    assert (x.ndim == y.ndim) and (x.ndim in (1, 2))
    if x.ndim == 1:
        x = x[:, None]
        y = y[:, None]
    assert x.shape[1] == y.shape[1]
    geom = PointCloud(x, y)
    problem = LinearProblem(geom)
    solver = UnivariateSolver()
    marginal_losses = jnp.asarray(solver(problem).ot_costs)
    return jnp.sqrt(jnp.mean(jnp.square(marginal_losses)))


class WassersteinDiscriminator(AbstractDiscriminator):
    bm_dim: int = eqx.field(static=True)

    def __init__(self, bm_dim):
        self.bm_dim = bm_dim

    def __call__(self, samples_true, samples_fake):
        # Either (x = hh, y = bb), or (x = w, y = la)
        x_true, y_true = samples_true
        x_fake, y_fake = samples_fake

        assert y_true.ndim == y_fake.ndim == x_true.ndim == x_fake.ndim == 2
        assert x_true.shape[1] == x_fake.shape[1] == self.bm_dim
        assert (
            y_true.shape[1] == y_fake.shape[1] == self.bm_dim * (self.bm_dim - 1) // 2
        )
        out = marginal_wass2_error(y_true, y_fake)
        out = eqxi.error_if(
            out,
            jnp.logical_not(jnp.allclose(x_true, x_fake)),
            "The marginal wasserstein discriminator should be used with a"
            " loss which has w_fake = w_true and hh_fake = hh_true, but the"
            " values are different.",
        )
        return out
