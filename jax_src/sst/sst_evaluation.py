import math
import os

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from jax_src.discriminator import marginal_wass2_error
from jax_src.generator import save_net
from jax_src.sst import SSTNet


def load_true_samples(file_name):
    with open(file_name, "rb") as f:
        w = np.load(f)
        hh = np.load(f)
        sst = np.load(f)

    return w, hh, sst


def wass2_errors(net: SSTNet, printing=False):
    dtype = net.dtype
    dir_name = "sst_saved_values"
    is_file = lambda f: os.path.isfile(os.path.join(dir_name, f))  # noqa: E731
    files = [os.path.join(dir_name, f) for f in os.listdir(dir_name) if is_file(f)]
    if len(files) > 20:
        files = files[:20]
    avg_error = jnp.array(0.0, dtype=dtype)
    for file_name in files:
        w, hh, c = load_true_samples(file_name)
        w = jnp.broadcast_to(jnp.array(w, dtype=dtype), (c.shape[0], 1))
        hh = jnp.broadcast_to(jnp.array(hh, dtype=dtype), (c.shape[0], 1))
        c_true = jnp.expand_dims(jnp.array(c, dtype=dtype), axis=1)
        c_fake = net.generate_c(jr.key(0), w, hh)

        error = marginal_wass2_error(c_true, c_fake)
        if printing:
            print(
                f"Error for w={float(w[0, 0]):.4}, hh={float(hh[0, 0]):.4}: {error:.4}"
            )
        avg_error += error
    avg_error /= len(files)
    return avg_error


def wass2_errors_normal(key, printing=False):
    dtype = jnp.float32
    dir_name = "sst_saved_values"
    is_file = lambda f: os.path.isfile(os.path.join(dir_name, f))  # noqa: E731
    files = [os.path.join(dir_name, f) for f in os.listdir(dir_name) if is_file(f)]
    if len(files) > 20:
        files = files[:20]
    avg_error = jnp.array(0.0, dtype=dtype)
    for file_name in files:
        w, hh, c = load_true_samples(file_name)
        w = float(w)
        hh = float(hh)
        c_true = jnp.array(c, dtype=dtype)
        mean, var = true_cond_stats_c(w, hh)
        c_fake = jr.normal(key, c_true.shape) * jnp.sqrt(var) + mean

        error = marginal_wass2_error(c_true, c_fake)
        if printing:
            print(f"Error for w={w:.4}, hh={hh:.4}: {error:.4}")
        avg_error += error
    avg_error /= len(files)
    return avg_error


def true_cond_stats_c(w, hh):
    w2 = w**2
    hh2 = hh**2
    mean = 1 / 3 * w2 + w * hh + 6 / 5 * hh2 + 1 / 15
    var = 11 / 6300 + 1 / 180 * w2 + 1 / 175 * hh2
    return mean, var


def stat_error(c: Array, w, hh):
    w = float(w)
    hh = float(hh)
    true_mean, true_var = true_cond_stats_c(w, hh)
    mean_error = jnp.abs(true_mean - jnp.mean(c))
    var_error = jnp.abs(true_var - jnp.var(c))
    return mean_error, var_error


def eval_net(
    net: SSTNet, key, num_reps, num_samples, runnning_score, printing=True, saving=False
):
    dtype = net.dtype
    key_w, key_hh, key_model = jr.split(key, 3)
    keys = jr.split(key_model, num_reps)
    mult = 3.0
    ws = mult * jr.normal(key_w, (num_reps,), dtype=dtype)
    hhs = mult * math.sqrt(1 / 12) * jr.normal(key_hh, (num_reps,), dtype=dtype)
    mean_errs = 0.0
    var_errs = 0.0
    avg_var = 0.0
    for i in range(num_reps):
        # print(f"Rep {i+1}/{num_reps}, w={float(ws[i]):.4}, hh={float(hhs[i]):.4}")
        w_broad = jnp.broadcast_to(ws[i], (num_samples,))
        hh_broad = jnp.broadcast_to(hhs[i], (num_samples,))
        c = net.generate_c(keys[i], w_broad, hh_broad)
        mean_error, var_error = stat_error(c, ws[i], hhs[i])
        fake_var = jnp.var(c)
        avg_var += fake_var
        mean_errs += mean_error
        var_errs += var_error
    # compute the average of the errors
    if num_reps > 0:
        mean_errs /= num_reps
        var_errs /= num_reps
        avg_var /= num_reps

    wass_err = wass2_errors(net)

    score = math.sqrt((mean_errs**2 + (10 * var_errs) ** 2 + (500 * wass_err) ** 2))
    if printing:
        print(
            f"Mean error: {mean_errs:.4}, variance error: {var_errs:.4}, avg var: {avg_var:.4}, wass error: {wass_err:.4}, score: {score:.4}"
        )

    if score < runnning_score:
        runnning_score = score
        print(f"New best net with score {runnning_score:.4}")
        if saving:
            save_net(net, "/home/andy/PycharmProjects/Levy_CFGAN/numpy_nets/sst_")
    elif runnning_score == -1:
        print(f"Inital score: {score:.4}")
        runnning_score = score

    return runnning_score, mean_errs, var_errs
