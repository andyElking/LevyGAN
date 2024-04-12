from functools import partial
from types import NoneType
from typing import Callable, TypeAlias, TypeVar

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
from jax import lax, Array
from jaxtyping import ArrayLike, Int, Float
import equinox as eqx
import equinox.internal as eqxi
import optax
from jax._src.random import KeyArray

from ..generator import AbstractNet
from ..discriminator import AbstractDiscriminator

DiscrType = TypeVar("DiscrType", AbstractDiscriminator, NoneType)
_LossType: TypeAlias = Callable[[tuple[AbstractNet, DiscrType], KeyArray], Array]
_CarryType = tuple[
    Int[ArrayLike, ""], AbstractNet, DiscrType, optax.OptState  # noqa: F722
]


def scan_step(
    carry: _CarryType,
    key,
    opt,
    loss: _LossType,
    lr_ratio: Float[ArrayLike, ""],  # noqa: F722
    num_discr_iters: Int[ArrayLike, ""],  # noqa: F722
):
    """A step of the training loop. The signature is set to conform to `lax.scan`."""
    itr, net, discriminator, opt_state = carry
    itr += 1
    holomorphic = (net.dtype == jnp.complex64) or (net.dtype == jnp.complex128)
    grad_loss = eqx.filter_value_and_grad(loss, holomorphic=holomorphic)
    loss_value, grad = grad_loss((net, discriminator), key)
    grad_net, grad_discriminator = grad

    # We want to maximize the loss of the discriminator
    # and minimize the loss of the generator
    grad_net = jtu.tree_map(lambda x: -x, grad_net)
    grad = (grad_net, grad_discriminator)

    params = (net, discriminator)
    updates, opt_state = opt.update(grad, opt_state, params)
    updates_net, updates_discr = updates

    # We don't update the net on each iteration
    def updated_net(_net):
        return jtu.tree_map(lambda x, y: x + y, _net, updates_net)

    # Make sure the predicate never gets batched using vmap
    pred = eqxi.nonbatchable(itr % num_discr_iters == 0)
    net = jax.lax.cond(pred, updated_net, lambda x: x, net)

    # The learning rate of the discriminator is greater than that of the generator
    # In addition we also clip the parameters of the discriminator
    def discr_update(param, update):
        out = param + update * lr_ratio
        # Maybe clip the parameters of the discriminator?
        # bound = 30.
        # out.real = jnp.clip(out.real, -bound, bound)
        # out.imag = jnp.clip(out.imag, -bound, bound)
        return out

    if discriminator is not None:
        discriminator = jtu.tree_map(discr_update, discriminator, updates_discr)

    carry = (itr, net, discriminator, opt_state)
    return carry, loss_value


@partial(
    jax.jit,
    static_argnames=["num_steps", "opt", "loss", "lr_ratio", "num_discr_iters"],
)
def train(
    net: AbstractNet,
    discriminator: DiscrType,
    key,
    num_steps: int,
    opt: optax.GradientTransformation,
    opt_state: optax.OptState,
    loss: _LossType,
    lr_ratio: float,
    num_discr_iters: int,
):
    keys = jr.split(key, num_steps)

    lr_ratio = jnp.asarray(lr_ratio, dtype=jnp.float32)
    num_discr_iters = jnp.asarray(num_discr_iters, dtype=jnp.int32)

    step_partial = partial(
        scan_step,
        opt=opt,
        loss=loss,
        lr_ratio=lr_ratio,
        num_discr_iters=num_discr_iters,
    )

    # The MAIN LOOP is performed using `lax.scan` for compilation performance
    init: _CarryType = (jnp.zeros((), dtype=jnp.int32), net, discriminator, opt_state)
    (itr, net, discriminator, opt_state), losses = lax.scan(step_partial, init, keys)
    return net, discriminator, opt_state, losses


def make_optimizer(net, discr, schedule, beta1, beta2):
    opt = optax.chain(
        optax.scale_by_adam(beta1, beta2),
        optax.scale_by_schedule(schedule),
    )
    opt_state = opt.init((net, discr))
    return opt, opt_state
