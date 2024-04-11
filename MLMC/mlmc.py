# Import Packages
import numpy as np
import torch
import os
from tqdm import tqdm
import sys
import pickle

sys.path.append(".")

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Import Functions
from mlmc_functions import do_mlmc, call_heston_cf
from src.model.Generator import PairNetGenerator


##################################################

##### Hyperparams of SDE and Exact Value #####

##################################################
np.random.seed(26)
coefs = {
    "T": 1,
    "r": 0.1,
    "k": 2,
    "theta": 0.1,
    "sigma": 0.5,
    "u0": np.log(20),
    "v0": 0.4,
}
R = coefs["r"]
U0 = coefs["u0"]
V0 = coefs["v0"]
K = 20
T = 1
payoff = lambda x, y: np.exp(-R * T) * np.maximum(0, (np.exp(x) - K))
# Exact value of call option
exact_val = call_heston_cf(
    s0=np.exp(U0),
    v0=V0,
    vbar=coefs["theta"],
    a=coefs["k"],
    vvol=coefs["sigma"],
    r=R,
    rho=0,
    t=T,
    k=K,
)

#################################################

#### Import NET ####

#################################################

gen_config = gen_config = {
    "use_pair_net": True,
    "bm_dim": 2,
    "noise_size": 4,  # size of latent space
    "num_layers": 3,
    "hidden_dim": 16,
    "activation": ("leaky", 0.01),  # or 'relu'
    "batch_norm": True,
    "pairnet_bn": True,
    "do_bridge_flipping": True,  # "bridge_flipping", otherwise off
    "use_mixed_noise": False,  # Uses noise from several distributions. Gives bad results for now...
    "use_attention": False,
    "enforce_antisym": False,
    "reinject_h": False,
    "gen_dict_saving_on": True,
}

generator = PairNetGenerator(gen_config)
generator.load_dict(
    filename="../good_model_saves/generator_4d_PairNet3LAY_16HID_lky0.01lky0.01ACT_BN_4noise_bf/gen_num5_best__scr.pt"
)


##################################################

##### Perform MLMCS #####

##################################################

# Define Timesteps and number of levels
num_levels = 5
levels = np.arange(num_levels)
milstein_dt0 = -1
milstein_dt = np.logspace(
    milstein_dt0, milstein_dt0 * num_levels, num=num_levels, base=2
)
strang_dt = 2 * milstein_dt
# Number of samples on each level
M_list = [2 ** (21 - i) for i in range(num_levels)]

# For each method perform MLMC for a number of iterations
num_iters = 40
# Output arrays
milstein_variances = np.zeros((num_iters, num_levels))
milstein_estimates = np.zeros((num_iters, num_levels))
milstein_anti_variances = np.zeros((num_iters, num_levels))
milstein_anti_estimates = np.zeros((num_iters, num_levels))
strang_net_variances = np.zeros((num_iters, num_levels))
strang_net_estimates = np.zeros((num_iters, num_levels))
strang_foster_variances = np.zeros((num_iters, num_levels))
strang_foster_estimates = np.zeros((num_iters, num_levels))
strang_rad_variances = np.zeros((num_iters, num_levels))
strang_rad_estimates = np.zeros((num_iters, num_levels))
strang_no_area_variances = np.zeros((num_iters, num_levels))
strang_no_area_estimates = np.zeros((num_iters, num_levels))

for i in tqdm(range(num_iters)):
    _, _, milstein_estimates[i], milstein_variances[i] = do_mlmc(
        coefs, payoff, M_list, milstein_dt, scheme="Milstein"
    )
    _, _, milstein_anti_estimates[i], milstein_anti_variances[i] = do_mlmc(
        coefs, payoff, M_list, milstein_dt, scheme="Milstein_anti"
    )
    _, _, strang_net_estimates[i], strang_net_variances[i] = do_mlmc(
        coefs, payoff, M_list, strang_dt, scheme="Strang", net=generator
    )
    _, _, strang_foster_estimates[i], strang_foster_variances[i] = do_mlmc(
        coefs, payoff, M_list, strang_dt, scheme="Strang_F"
    )
    _, _, strang_rad_estimates[i], strang_rad_variances[i] = do_mlmc(
        coefs, payoff, M_list, strang_dt, scheme="Strang_R"
    )
    _, _, strang_no_area_estimates[i], strang_no_area_variances[i] = do_mlmc(
        coefs, payoff, M_list, strang_dt, scheme="Strang_NA"
    )

# Record mean of the arrays across iterations
milstein_errors = np.abs(np.cumsum(milstein_estimates, axis=1) - exact_val)
milstein_estimate_mean = np.mean(milstein_errors, axis=0)
milstein_variance_mean = np.mean(milstein_variances, axis=0)
milstein_anti_errors = np.abs(np.cumsum(milstein_anti_estimates, axis=1) - exact_val)
milstein_anti_estimate_mean = np.mean(milstein_anti_errors, axis=0)
milstein_anti_variance_mean = np.mean(milstein_anti_variances, axis=0)
strang_net_errors = np.abs(np.cumsum(strang_net_estimates, axis=1) - exact_val)
strang_net_estimate_mean = np.mean(strang_net_errors, axis=0)
strang_net_variance_mean = np.mean(strang_net_variances, axis=0)
strang_foster_errors = np.abs(np.cumsum(strang_foster_estimates, axis=1) - exact_val)
strang_foster_estimate_mean = np.mean(strang_foster_errors, axis=0)
strang_foster_variance_mean = np.mean(strang_foster_variances, axis=0)
strang_rad_errors = np.abs(np.cumsum(strang_rad_estimates, axis=1) - exact_val)
strang_rad_estimate_mean = np.mean(strang_rad_errors, axis=0)
strang_rad_variance_mean = np.mean(strang_rad_variances, axis=0)
strang_no_area_errors = np.abs(np.cumsum(strang_no_area_estimates, axis=1) - exact_val)
strang_no_area_estimate_mean = np.mean(strang_no_area_errors, axis=0)
strang_no_area_variance_mean = np.mean(strang_no_area_variances, axis=0)

# Estimate gradients
strang_net_variance_grad = np.round(
    (
        np.emath.logn(2, strang_net_variance_mean[-1])
        - np.emath.logn(2, strang_net_variance_mean[1])
    )
    / (levels[-1] - levels[1]),
    2,
)
strang_no_area_variance_grad = np.round(
    (
        np.emath.logn(2, strang_no_area_variance_mean[-1])
        - np.emath.logn(2, strang_no_area_variance_mean[1])
    )
    / (levels[-1] - levels[1]),
    2,
)
milstein_estimate_grad = np.round(
    (
        np.emath.logn(2, milstein_estimate_mean[-1])
        - np.emath.logn(2, milstein_estimate_mean[1])
    )
    / (levels[-1] - levels[1]),
    2,
)
strang_net_estimate_grad = np.round(
    (
        np.emath.logn(2, strang_net_estimate_mean[-2])
        - np.emath.logn(2, strang_net_estimate_mean[0])
    )
    / (levels[-2] - levels[0]),
    2,
)
strang_no_area_estimate_grad = np.round(
    (
        np.emath.logn(2, strang_no_area_estimate_mean[-2])
        - np.emath.logn(2, strang_no_area_estimate_mean[0])
    )
    / (levels[-2] - levels[0]),
    2,
)
# Write results to pickle file
results = {
    "milstein_estimate_mean": milstein_estimate_mean,
    "milstein_variance_mean": milstein_variance_mean,
    "milstein_anti_estimate_mean": milstein_anti_estimate_mean,
    "milstein_anti_variance_mean": milstein_anti_variance_mean,
    "strang_net_estimate_mean": strang_net_estimate_mean,
    "strang_net_variance_mean": strang_net_variance_mean,
    "strang_foster_estimate_mean": strang_foster_estimate_mean,
    "strang_foster_variance_mean": strang_foster_variance_mean,
    "strang_rad_estimate_mean": strang_rad_estimate_mean,
    "strang_rad_variance_mean": strang_rad_variance_mean,
    "strang_no_area_estimate_mean": strang_no_area_estimate_mean,
    "strang_no_area_variance_mean": strang_no_area_variance_mean,
    "milstein_estimate_grad": milstein_estimate_grad,
    "strang_net_estimate_grad": strang_net_estimate_grad,
    "strang_no_area_estimate_grad": strang_no_area_estimate_grad,
    "strang_net_variance_grad": strang_net_variance_grad,
    "strang_no_area_variance_grad": strang_no_area_variance_grad,
}
with open("MLMC/MLMC_results/mlmc_results.pkl", "wb+") as f:
    pickle.dump(results, f)
