# Plotting
### Plotting ###
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from matplotlib import rc

# Other imports
import numpy as np
import pickle
import sys

sys.path.append(".")

rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
rc("text", usetex=True)
plt.rcParams.update(
    {"text.usetex": True, "text.latex.preamble": r"\usepackage{amsfonts}"}
)
colors = ["#60ceaf", "#de76bc", "#81cb61", "#f90031", "#8449be", "#5ca9ff"]

# Extract results from pickle file
results = pickle.load(open("MLMC/MLMC_results/mlmc_results.pkl", "rb"))
results_anti = pickle.load(open("MLMC/MLMC_results/mlmc_strang_anti.pkl", "rb"))
milstein_estimate_mean = results["milstein_estimate_mean"]
milstein_variance_mean = results["milstein_variance_mean"]
milstein_anti_estimate_mean = results["milstein_anti_estimate_mean"]
milstein_anti_variance_mean = results["milstein_anti_variance_mean"]
strang_net_estimate_mean = results["strang_net_estimate_mean"]
strang_net_variance_mean = results["strang_net_variance_mean"]
strang_foster_estimate_mean = results["strang_foster_estimate_mean"]
strang_foster_variance_mean = results["strang_foster_variance_mean"]
strang_rad_estimate_mean = results["strang_rad_estimate_mean"]
strang_rad_variance_mean = results["strang_rad_variance_mean"]
strang_no_area_estimate_mean = results["strang_no_area_estimate_mean"]
strang_no_area_variance_mean = results["strang_no_area_variance_mean"]
strang_no_area_anti_estimate_mean = results_anti["strang_anti_estimate_mean"]
strang_no_area_anti_variance_mean = results_anti["strang_anti_variance_mean"]
milstein_estimate_grad = results["milstein_estimate_grad"]
strang_net_estimate_grad = results["strang_net_estimate_grad"]
strang_no_area_estimate_grad = results["strang_no_area_estimate_grad"]
strang_net_variance_grad = results["strang_net_variance_grad"]
strang_no_area_variance_grad = results["strang_no_area_variance_grad"]
levels = np.array([0, 1, 2, 3, 4])
##################################################

##### AREA VS NO AREA #####

##################################################


############# Plot variances on a log 2 scale #############
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# Set y-axis to log 2 scale
ax.set_yscale("log", base=2)
# Plot the variances
ax.plot(
    levels,
    milstein_variance_mean,
    label="Milstein",
    color=colors[0],
    marker="^",
    linestyle=":",
    linewidth=2,
)
ax.plot(
    levels,
    milstein_anti_variance_mean,
    label="Milstein-Anti",
    color=colors[1],
    marker="v",
    linestyle="--",
    linewidth=2,
)
ax.plot(
    levels,
    strang_net_variance_mean,
    label="Strang-Net",
    color=colors[2],
    marker="s",
    linewidth=2,
)
# Plot straight line estimating gradient of Strang-Net
ax.plot(
    levels,
    strang_net_variance_mean[1] * 2 ** (strang_net_variance_grad * (levels - 1)),
    label="Gradient " + str(strang_net_variance_grad),
    color="black",
    linestyle="-.",
    linewidth=1.2,
)
# Plot parameters
ax.tick_params(axis="both", labelsize=15)
ax.yaxis.label.set_size(22)
ax.xaxis.label.set_size(22)
ax.set_xlabel("Level")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_ylabel("Variance")
ax.legend(fontsize=12, loc="upper right")
fig.suptitle("Multilevel Variance Comparison", fontsize=25)
fig.tight_layout()
plt.savefig("fake_vs_no_fake_variance.png")

############# Plot estimates on a log 2 scale #############
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# Set y-axis to log 2 scale
ax.set_yscale("log", base=2)
# Plot the estimates
ax.plot(
    levels,
    milstein_estimate_mean,
    label="Milstein",
    color=colors[0],
    marker="^",
    linestyle=":",
    linewidth=2,
)
ax.plot(
    levels,
    milstein_anti_estimate_mean,
    label="Milstein-Anti",
    color=colors[1],
    marker="v",
    linestyle="--",
    linewidth=2,
)
ax.plot(
    levels,
    strang_net_estimate_mean,
    label="Strang-Net",
    color=colors[2],
    marker="s",
    linewidth=2,
)
# Plot straight line estimating gradient of Strang-Net and Milstein
ax.plot(
    levels,
    strang_net_estimate_mean[0] * 2 ** (strang_net_estimate_grad * levels),
    label="Gradient " + str(strang_net_estimate_grad),
    color="black",
    linestyle="-.",
    linewidth=1.2,
)
ax.plot(
    levels,
    milstein_estimate_mean[1] * 2 ** (milstein_estimate_grad * (levels - 1)),
    label="Gradient " + str(milstein_estimate_grad),
    color="steelblue",
    linestyle="-.",
    linewidth=1.2,
)
# Plot parameters
ax.tick_params(axis="both", labelsize=15)
ax.yaxis.label.set_size(22)
ax.xaxis.label.set_size(22)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xlabel("Level")
ax.set_ylabel("Error")
ax.legend(fontsize=12, loc="upper right")
fig.suptitle("Multilevel Error Comparison", fontsize=25)
fig.tight_layout()
plt.savefig("fake_vs_no_fake_error.png")


##################################################

##### COMPARE AREAS #####

##################################################

############# Plot variances on a log 2 scale #############
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# Set y-axis to log 2 scale
ax.set_yscale("log", base=2)
# Plot the variances
ax.plot(
    levels,
    strang_net_variance_mean,
    label="Strang-Net",
    color=colors[2],
    linewidth=2,
    marker="s",
)
ax.plot(
    levels,
    strang_rad_variance_mean,
    label="Strang-T",
    color=colors[3],
    marker="^",
    linestyle="--",
    linewidth=2,
)
ax.plot(
    levels,
    strang_foster_variance_mean,
    label="Strang-F",
    color=colors[4],
    marker="v",
    linestyle=":",
    linewidth=2,
)
ax.plot(
    levels,
    strang_no_area_variance_mean,
    label="Strang-NA",
    color=colors[5],
    marker="o",
    linestyle="-",
    linewidth=2,
)
ax.plot(
    levels,
    strang_no_area_anti_variance_mean,
    label="Strang-Anti",
    color="darkorange",
    marker="<",
    linestyle="dotted",
    linewidth=2,
)
# Plot straight line estimating gradient of Strang-Net
ax.plot(
    levels,
    strang_net_variance_mean[1] * 2 ** (strang_net_variance_grad * (levels - 1)),
    label="Gradient " + str(strang_net_variance_grad),
    color="black",
    linestyle="-.",
    linewidth=1.2,
)
# Plot straight line estimating gradient of Strang-Net
ax.plot(
    levels,
    strang_no_area_variance_mean[1]
    * 2 ** (strang_no_area_variance_grad * (levels - 1)),
    label="Gradient " + str(strang_no_area_variance_grad),
    color="steelblue",
    linestyle="-.",
    linewidth=1.2,
)
# Plot parameters
ax.tick_params(axis="both", labelsize=15)
ax.yaxis.label.set_size(22)
ax.xaxis.label.set_size(22)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xlabel("Level")
ax.set_ylabel("Variance")
ax.legend(fontsize=12, loc="lower left")
fig.suptitle("Multilevel Variance Comparison", fontsize=25)
fig.tight_layout()
plt.savefig("areas_variance.png")

############# Plot estimates on a log 2 scale #############
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# Set y-axis to log 4 scale
ax.set_yscale("log", base=2)
# Plot the estimates
ax.plot(
    levels,
    strang_net_estimate_mean,
    label="Strang-Net",
    color=colors[2],
    linewidth=2,
    marker="s",
)
ax.plot(
    levels,
    strang_rad_estimate_mean,
    label="Strang-T",
    color=colors[3],
    marker="^",
    linestyle="--",
    linewidth=2,
)
ax.plot(
    levels,
    strang_foster_estimate_mean,
    label="Strang-F",
    color=colors[4],
    marker="v",
    linestyle=":",
    linewidth=2,
)
ax.plot(
    levels,
    strang_no_area_estimate_mean,
    label="Strang-NA",
    color=colors[5],
    marker="o",
    linestyle="-",
    linewidth=2,
)
ax.plot(
    levels,
    strang_no_area_anti_estimate_mean,
    label="Strang-Anti",
    color="darkorange",
    marker="<",
    linestyle="dotted",
    linewidth=2,
)
# Plot straight line estimating gradient of Strang-Net
ax.plot(
    levels,
    strang_net_estimate_mean[0] * 2 ** (strang_net_estimate_grad * levels),
    label="Gradient " + str(strang_net_estimate_grad),
    color="black",
    linestyle="-.",
    linewidth=1.2,
)
ax.plot(
    levels,
    strang_no_area_estimate_mean[0] * 2 ** (strang_no_area_estimate_grad * levels),
    label="Gradient " + str(strang_no_area_estimate_grad),
    color="steelblue",
    linestyle="-.",
    linewidth=1.2,
)
# Plot parameters
ax.tick_params(axis="both", labelsize=15)
ax.yaxis.label.set_size(22)
ax.xaxis.label.set_size(22)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xlabel("Level")
ax.set_ylabel("Error")
ax.legend(fontsize=12, loc="lower left")
fig.suptitle("Multilevel Error Comparison", fontsize=25)
fig.tight_layout()
plt.savefig("areas_error.png")
