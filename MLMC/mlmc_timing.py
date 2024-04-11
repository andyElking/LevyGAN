# In this file, we implement the MLMC algorithm for the Heston model using the Milstein and Strang splitting schemes. We also implement the optimal number of levels and samples for the MLMC algorithm.

# Import packages
import numpy as np
import torch
from math import sqrt
import time
from tqdm import tqdm
import pickle
import sys

sys.path.append("..")

# Import functions
from mlmc_functions import do_mlmc_giles, call_heston_cf
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

#################################################

#### Timing tests ####

#################################################

# For each desired level of accuracy, we determine the optimal number of levels and samples for the MLMC algorithm. We then compare the time taken to run the MLMC algorithm for the Heston model using the Milstein and Strang splitting schemes.

# Number of accuracies to test
num_accuracies = 10
accuracies = np.logspace(-1, -2.3, num_accuracies)  #
# Burn in length for timing and number of iterations
burn_in = 5
num_iters = 30
# Initialise outputs
time_milstein = np.zeros(num_accuracies)
time_strang = np.zeros(num_accuracies)
# Store all errors and variances
errors_milstein = np.zeros((num_accuracies, num_iters))
errors_strang = np.zeros((num_accuracies, num_iters))
variances_milstein = np.zeros((num_accuracies, num_iters))
variances_strang = np.zeros((num_accuracies, num_iters))


# Loop over accuracies, and perform timing for each iteration
for i in tqdm(range(num_accuracies)):
    # Current accuracy
    accuracy = accuracies[i]
    # Loop over iterations
    milstein_time = 0
    strang_time = 0
    for j in range(num_iters):
        # Time Milstein
        start = time.time()
        milstein_estimate, milstein_variance = do_mlmc_giles(
            coefs, payoff, dt0=(1 / 2), scheme="Milstein", accuracy=accuracy
        )
        end = time.time()
        if j >= burn_in:
            milstein_time += end - start

        # Time Strang
        start = time.time()
        strang_estimate, strang_variance = do_mlmc_giles(
            coefs, payoff, dt0=1, scheme="Strang", net=generator, accuracy=accuracy
        )
        end = time.time()
        if j >= burn_in:
            strang_time += end - start

        # Errors
        errors_milstein[i, j] = np.abs(milstein_estimate - exact_val)
        errors_strang[i, j] = np.abs(strang_estimate - exact_val)
        # Variances
        variances_milstein[i, j] = milstein_variance
        variances_strang[i, j] = strang_variance

    # Average time
    time_milstein[i] = milstein_time / (num_iters - burn_in)
    time_strang[i] = strang_time / (num_iters - burn_in)

rmse_milstein = np.sqrt(np.mean(errors_milstein**2, axis=1))
rmse_strang = np.sqrt(np.mean(errors_strang**2, axis=1))


# Write results to pickle file
results = {
    "accuracies": accuracies,
    "time_milstein": time_milstein,
    "time_strang": time_strang,
    "errors_milstein": errors_milstein,
    "errors_strang": errors_strang,
    "variances_milstein": variances_milstein,
    "variances_strang": variances_strang,
    "rmse_milstein": rmse_milstein,
    "rmse_strang": rmse_strang,
}

with open("MLMC/MLMC_results/mlmc_timing.pkl", "wb+") as f:
    pickle.dump(results, f)
