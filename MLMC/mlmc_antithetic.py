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
M_list = [2 ** (26 - i) for i in range(num_levels)]

# For each method perform MLMC for a number of iterations
num_iters = 40
# Output arrays
strang_anti_variances = np.zeros((num_iters, num_levels))
strang_anti_estimates = np.zeros((num_iters, num_levels))

for i in tqdm(range(num_iters)):
    _, _, strang_anti_estimates[i], strang_anti_variances[i] = do_mlmc(
        coefs, payoff, M_list, strang_dt, scheme="Strang_NA_anti"
    )

# Record mean of the arrays across iterations
strang_anti_errors = np.abs(np.cumsum(strang_anti_estimates, axis=1) - exact_val)
strang_anti_estimate_mean = np.mean(strang_anti_errors, axis=0)
strang_anti_variance_mean = np.mean(strang_anti_variances, axis=0)

# Write results to pickle file
results = {
    "strang_anti_estimate_mean": strang_anti_estimate_mean,
    "strang_anti_variance_mean": strang_anti_variance_mean,
}
with open("MLMC/MLMC_results/mlmc_strang_anti.pkl", "wb+") as f:
    pickle.dump(results, f)
