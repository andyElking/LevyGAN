import math
from functools import partial

import jax
import numpy as np
import ot
from jax import Array, numpy as jnp, random as jr

from jax_src.generator import generate_la
from jax_src.aux_functions import la_chen
from jax_src.fosters_method import mom4_jax


def _indices(order: int, start: int, stop: int) -> tuple[np.ndarray, ...]:
    if order == 1:
        return (np.arange(start, stop),)

    elif order == 2:
        one, two = np.triu_indices(stop - start, k=0)
        return one + start, two + start

    elif order > 2:
        new_arr_list = []
        lower_order_lists = tuple([[] for _ in range(order - 1)])
        for i in range(start, stop):
            lower_ord = _indices(order - 1, i, stop)
            for j, arr in enumerate(lower_ord):
                lower_order_lists[j].append(arr)
            new_arr_list.append(np.full_like(lower_ord[0], i))
        list_of_outputs = [np.concatenate(new_arr_list)]
        for list_of_arrs in lower_order_lists:
            list_of_outputs.append(np.concatenate(list_of_arrs))
        return tuple(list_of_outputs)


def nth_moment_indices(d, n):
    """Compute indices using which one can compute all nth cross-moments of
    a d-dimensional random variable."""
    return _indices(n, 0, d)


def empirical_nth_moments(samples: Array, n: int) -> Array:
    assert samples.ndim == 2
    dim = samples.shape[1]
    indices = nth_moment_indices(dim, n)
    num_moments = indices[0].shape[0]
    num_samples = samples.shape[0]

    @jax.jit
    def compute_fast(samples, indices):
        init = jnp.ones((num_samples, num_moments), dtype=samples.dtype)
        indices_arr = jnp.stack(indices, axis=0)

        def step(carry, idx):
            return carry * samples[:, idx], None

        result, _ = jax.lax.scan(step, init, indices_arr)
        return jnp.mean(result, axis=0)

    return compute_fast(samples, indices)


def errors(x: Array, y: Array) -> tuple[Array, Array, Array]:
    """Computes the mean, max and RMS error between `x` and `y`."""
    assert x.shape == y.shape
    abs_diff = jnp.abs(x - y)
    mean = jnp.mean(abs_diff)
    max = jnp.max(abs_diff)
    rms = jnp.sqrt(jnp.mean(abs_diff**2))
    return mean, max, rms


@partial(jax.jit, static_argnums=(0, 1, 2))
def _split_test(test_fun, total_n, batch_n, key):
    """A helper which splits the test into smaller batches and averages the results.
    For example, we need 2**20 samples to compute 4th moments accurately, but having
    all those samples in memory at once is not feasible, so we split into smaller
    batches."""
    num_batches = total_n // batch_n
    keys = jr.split(key, num_batches)

    def step(_: None, k):
        return None, test_fun(k, batch_n)

    _, results = jax.lax.scan(step, None, keys, length=num_batches)

    return jnp.mean(results, axis=0)


vec_la_chen = jax.jit(jax.vmap(la_chen, in_axes=(0, 0, 0, 0)))


@partial(jax.jit, static_argnames=("gen_la_fun", "num_samples", "bm_dim", "dtype"))
def _wass2_chen_helper(key, gen_la_fun, num_samples, bm_dim, dtype):
    key_w, key_la_true, key_la_fake = jr.split(key, 3)
    keys_w = jr.split(key_w, 4)
    keys_la_true = jr.split(key_la_true, 4)
    ws = []
    las = []
    for i in range(4):
        w = jr.normal(keys_w[i], (num_samples, bm_dim), dtype=dtype)
        la = gen_la_fun(keys_la_true[i], w)
        # _, _, la = generate_la(keys_la_true[i], net, triu_indices, w, None)
        ws.append(w)
        las.append(la)
    w12, la12 = vec_la_chen(ws[0], las[0], ws[1], las[1])
    w34, la34 = vec_la_chen(ws[2], las[2], ws[3], las[3])
    del ws, las
    w_true, la_true = vec_la_chen(w12, la12, w34, la34)
    del w12, la12, w34, la34
    la_fake = gen_la_fun(key_la_fake, w_true)
    return la_true, la_fake


def wass2_chen_err(key, gen_la_fun, num_samples, bm_dim, dtype):
    """Computes the Wasserstein-2 error between the  samples
    generated by the net and the twice chen-combined samples
    (also generated by the net)."""
    la_true, la_fake = _wass2_chen_helper(key, gen_la_fun, num_samples, bm_dim, dtype)
    la_true_np = np.array(jnp.real(la_true), dtype=np.float64)
    la_fake_np = np.array(jnp.real(la_fake), dtype=np.float64)
    marg_wass_dist = math.sqrt(ot.wasserstein_1d(la_true_np, la_fake_np, p=2).item())
    return marg_wass_dist


def evaluate_method(
    gen_la_fun,
    key,
    true_moments,
    num_samples: int,
    bm_dim: int,
    dtype,
    print_results: bool = True,
):
    """Evaluates the net by computing various moments and comparing them to the true
    moments of Levy area (not conditioned on W)."""
    triu_len = bm_dim * (bm_dim - 1) // 2

    def get_moment_test_fun(n_moment):
        def test_fun(key, n_samples):
            key_w, key_la = jr.split(key, 2)
            w = jr.normal(key_w, (n_samples, bm_dim))
            la = gen_la_fun(key_la, w)
            return empirical_nth_moments(la, n_moment)

        return test_fun

    emp_4mom = _split_test(get_moment_test_fun(4), num_samples, 2**16, key)
    errs_mom4_mean, errs_mom4_max, errs_mom4_rms = errors(true_moments, emp_4mom)

    # All third moments should be 0
    emp_3mom = _split_test(get_moment_test_fun(3), num_samples, 2**16, key)
    errs_mom3_mean, errs_mom3_max, errs_mom3_rms = errors(
        jnp.zeros_like(emp_3mom), emp_3mom
    )

    # Covariance should be 0.25 * eye
    true_cov = jnp.eye(triu_len) * 0.25

    def cov_test_fun(key, n_samples):
        key_w, key_la = jr.split(key, 2)
        w = jr.normal(key_w, (n_samples, bm_dim))
        la = gen_la_fun(key_la, w)
        return jnp.cov(la, rowvar=False)

    emp_cov = _split_test(cov_test_fun, num_samples, 2**16, key)
    errs_cov_mean, errs_cov_max, errs_cov_rms = errors(true_cov, emp_cov)

    # Means should be 0
    true_means = jnp.zeros(triu_len)

    def means_test_fun(key, n_samples):
        key_w, key_la = jr.split(key, 2)
        w = jr.normal(key_w, (n_samples, bm_dim))
        la = gen_la_fun(key_la, w)
        return jnp.mean(la, axis=0)

    emp_means = _split_test(means_test_fun, num_samples, 2**16, key)
    errs_means_mean, errs_means_max, errs_means_rms = errors(true_means, emp_means)

    # Mean of absolute values should be 0.3712268722041659
    # See bottom of https://fabricebaudoin.blog/2016/06/30/lecture-1-the-paul-levys-stochastic-area-formula/
    true_mean_abs = 0.3712268722 * jnp.ones((triu_len,))

    def mean_abs_test_fun(key, n_samples):
        key_w, key_la = jr.split(key, 2)
        w = jr.normal(key_w, (n_samples, bm_dim))
        la = gen_la_fun(key_la, w)
        return jnp.mean(jnp.abs(la), axis=0)

    emp_mean_abs = _split_test(mean_abs_test_fun, num_samples, 2**16, key)
    errs_abs_mean, errs_abs_max, errs_abs_rms = errors(true_mean_abs, emp_mean_abs)

    wass2_err = wass2_chen_err(key, gen_la_fun, num_samples, 2, dtype)

    if print_results:
        print(
            f"Wass2: {wass2_err:.3}; MOMS: 4max: {errs_mom4_max:.3}, 4avg: {errs_mom4_mean:.3}, 3max: {errs_mom3_max:.3}, 2max: {errs_cov_max:.3}, 1max: {errs_means_max:.3}, 0max: {errs_abs_max:.3}"
        )
        # print(
        #     f"4th moment errors: mean: {errs_mom4_mean:.3}, "
        #     f"max: {errs_mom4_max:.3}, rms: {errs_mom4_rms:.3}\n"
        #     f"3rd moment errors: mean: {errs_mom3_mean:.3}, "
        #     f"max: {errs_mom3_max:.3}, rms: {errs_mom3_rms:.3}\n"
        #     f"Covariance errors: mean: {errs_cov_mean:.3}, "
        #     f"max: {errs_cov_max:.3}, rms: {errs_cov_rms:.3}\n"
        #     f"Mean       errors: mean: {errs_means_mean:.3}, "
        #     f"max: {errs_means_max:.3}, rms: {errs_means_rms:.3}\n"
        #     f"Mean(abs)  errors: mean: {errs_abs_mean:.3}, "
        #     f"max: {errs_abs_max:.3}, rms: {errs_abs_rms:.3}\n"
        # )

    return (
        (errs_mom4_mean, errs_mom4_max, errs_mom4_rms),
        (errs_mom3_mean, errs_mom3_max, errs_mom3_rms),
        (errs_cov_mean, errs_cov_max, errs_cov_rms),
        (errs_means_mean, errs_means_max, errs_means_rms),
        (errs_abs_mean, errs_abs_max, errs_abs_rms),
        wass2_err,
    )


# Evaluate the net


def _make_net_gen_la_fun(net):
    def gen_la_fun(key, w):
        triu_indices = jnp.triu_indices(w.shape[1], k=1)
        _, _, la = generate_la(key, net, triu_indices, w, None)
        return la

    return gen_la_fun


def evaluate_net(net, key, true_moments, num_samples, bm_dim):
    dtype = net.dtype
    gen_la_fun = _make_net_gen_la_fun(net)
    return evaluate_method(gen_la_fun, key, true_moments, num_samples, bm_dim, dtype)


# Evaluate Foster's method
def _gen_la_fun_foster(key, w):
    triu_indices = jnp.triu_indices(w.shape[1], k=1)
    num_samples = w.shape[0]
    keys = jr.split(key, num_samples)
    _, _, _, la = mom4_jax(keys, triu_indices, w, None, None)
    return la


def evaluate_fosters_method(key, true_moments, num_samples, bm_dim):
    return evaluate_method(
        _gen_la_fun_foster, key, true_moments, num_samples, bm_dim, jnp.float64
    )
