from equinox import AbstractVar
from jaxtyping import PyTree, Array
import equinox as eqx


class AbstractDiscriminator(eqx.Module):
    bm_dim: AbstractVar[int]

    def __call__(
        self, samples_true: PyTree[Array], samples_fake: PyTree[Array]
    ) -> Array:
        """Computes the loss of the discriminator.

        **Arguments:**
            - `samples_true`: `PyTree` of `jax.Array`
            - `samples_fake`: `PyTree` of `jax.Array`

        **Returns:**
            jax.Array of shape `()`
        """
        raise NotImplementedError
