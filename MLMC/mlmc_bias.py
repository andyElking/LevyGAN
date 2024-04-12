import numpy as np
import torch
import os
from tqdm import tqdm
import sys
import pickle

sys.path.append(".")

# Import Functions
from mlmc_functions import levy_bias_estimator
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


# Estimate Bias of using fake Levy area on each level
num_runs = 2**8
M = 2**20
dt = np.logspace(0, -4, num=5, base=2)
levy_bias = np.zeros((num_runs, len(dt) - 1))
for i in tqdm(range(num_runs)):
    for j in range(len(dt) - 1):
        levy_bias[i, j] = levy_bias_estimator(coefs, payoff, M, dt[j], generator)
levy_bias = np.mean(levy_bias, axis=0)

# Save results in pickle file
results = {
    "levy_bias": levy_bias,
}
pickle.dump(results, open("MLMC/MLMC_results/levy_bias_results.pkl", "wb+"))
