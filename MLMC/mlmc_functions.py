# In this file, we implement the MLMC algorithm for the Heston model using the Milstein and Strang splitting schemes. We also implement the optimal number of levels and samples for the MLMC algorithm.

# Import packages
import numpy as np
from scipy import integrate
import torch
from math import sqrt
import time
import pickle
from src.aux_functions import MC_chen_combine, combine_increments, mlmc_foster, mlmc_rad

device = "cuda" if torch.cuda.is_available() else "cpu"

##############################################################################################

## functions for calculating the exact value of Heston Call Option ##
## these are written based on the formula and Matlab code from Crisóstomo (2014) ##
## https://doi.org/10.48550/arXiv.1502.02963 ##

##############################################################################################


def chfun_heston(s0, v0, vbar, a, vvol, r, rho, t, w):
    # Heston characteristic function.
    # Inputs:
    # s0: stock price
    # v0: initial volatility (v0^2 initial variance)
    # vbar: long-term variance mean
    # a: variance mean-reversion speed
    # vvol: volatility of the variance process
    # r : risk-free rate
    # rho: correlation between the Weiner processes for the stock price and its variance
    # w: points at which to evaluate the function
    # Output:
    # Characteristic function of log (St) in the Heston model
    # Interim calculations
    alpha = -w * w / 2 - 1j * w / 2
    beta = a - rho * vvol * 1j * w
    gamma = vvol * vvol / 2
    h = np.sqrt(beta * beta - 4 * alpha * gamma)
    rplus = (beta + h) / vvol / vvol
    rminus = (beta - h) / vvol / vvol
    g = rminus / rplus
    # Required inputs for the characteristic function
    C = a * (rminus * t - (2 / vvol**2) * np.log((1 - g * np.exp(-h * t)) / (1 - g)))
    D = rminus * (1 - np.exp(-h * t)) / (1 - g * np.exp(-h * t))
    # Characteristic function evaluated at points w
    y = np.exp(C * vbar + D * v0 + 1j * w * np.log(s0 * np.exp(r * t)))
    return y


def call_heston_cf(s0, v0, vbar, a, vvol, r, rho, t, k):
    # Heston call value using characteristic functions.
    # y = call_heston_cf(s0, v0, vbar, a, vvol, r, rho, t, k)
    # Inputs:
    # s0: stock price
    # v0: initial volatility (v0^2 initial variance)
    # vbar: long-term variance mean
    # a: variance mean-reversion speed
    # vvol: volatility of the variance process
    # r: risk-free rate
    # rho: correlation between the Weiner processes of the stock price and its variance
    # t: time to maturity
    # k: option strike
    # chfun_heston: Heston characteristic function
    # 1st step: calculate pi1 and pi2
    # Inner integral 1
    int1 = lambda w, s0, v0, vbar, a, vvol, r, rho, t, k: np.real(
        np.exp(-1j * w * np.log(k))
        * chfun_heston(s0, v0, vbar, a, vvol, r, rho, t, w - 1j)
        / (1j * w * chfun_heston(s0, v0, vbar, a, vvol, r, rho, t, -1j))
    )  # inner integral1
    int1_val, tol1 = integrate.quad(
        lambda w: int1(w, s0, v0, vbar, a, vvol, r, rho, t, k), 0, 100
    )
    # numerical integration
    pi1 = int1_val / np.pi + 0.5  # final pi1
    # Inner integral 2:
    int2 = lambda w, s0, v0, vbar, a, vvol, r, rho, t, k: np.real(
        np.exp(-1j * w * np.log(k))
        * chfun_heston(s0, v0, vbar, a, vvol, r, rho, t, w)
        / (1j * w)
    )
    int2_val, tol2 = integrate.quad(
        lambda w: int2(w, s0, v0, vbar, a, vvol, r, rho, t, k), 0, 100
    )
    int2_val = np.real(int2_val)
    pi2 = int2_val / np.pi + 0.5  # final pi2
    # 3rd step: calculate call value
    y = s0 * pi1 - np.exp(-r * t) * k * pi2

    return y


##################################################

##### One Step Numerical Schemes #####

##################################################


def milstein(dw1, dw2, dt, u, v, coefs):
    """Performs one step of the Milstein no area scheme

    Args:
        dw1: increment of W_1
        dw2: increment of W_2
        dt: time step
        u: initial value of u
        v: initial value of v
        coefs: parameters of SDE

    Returns:
        : updated u and v
    """
    # function specifying the Milstein no area scheme
    u_new = (
        u
        + (coefs["r"] - 0.5 * v) * dt
        + np.sqrt(np.maximum(0, v)) * dw1
        + 0.25 * coefs["sigma"] * dw1 * dw2
    )
    v_new = (
        v
        + coefs["k"] * (coefs["theta"] - v) * dt
        + coefs["sigma"] * np.sqrt(np.maximum(0, v)) * dw2
        + 0.25 * coefs["sigma"] ** 2 * (dw2**2 - dt)
    )

    return u_new, v_new


def strang_splitting(dw1, dw2, a12, dt, u, v, coefs):
    """Performs one step of the Strang log-ODE scheme

    Args:
        dw1: increment of W_1
        dw2: increment of W_2
        a12: increment of levy area
        dt: time step
        u: initial value of u
        v: initial value of v
        coefs: parameters of SDE

    Returns:
        : updated u and v
    """
    xi = coefs["theta"] - coefs["sigma"] ** 2 / (4 * coefs["k"])
    sig = coefs["sigma"]
    kappa = coefs["k"]
    r = coefs["r"]

    # solving drift to half timestep
    v_d = xi + (v - xi) * np.exp(-0.5 * kappa * dt)
    u_d = (
        u
        + 0.5 * dt * (r - 0.5 * xi)
        + 0.5 * (1 / kappa) * (v - xi) * (np.exp(-0.5 * kappa * dt) - 1)
    )
    # solving diffusion
    v_dd = (np.sqrt(v_d) + 0.5 * sig * dw2) ** 2
    u_dd = u_d + np.sqrt(v_d) * dw1 + 0.25 * sig * dw1 * dw2 - 0.5 * sig * a12
    # solving drift 2nd half timestep
    v_new = xi + (v_dd - xi) * np.exp(-0.5 * kappa * dt)
    u_new = (
        u_dd
        + 0.5 * dt * (r - 0.5 * xi)
        + 0.5 * (1 / kappa) * (v_dd - xi) * (np.exp(-0.5 * kappa * dt) - 1)
    )

    return u_new, v_new


#################################################

#### MLMC Schemes per Level ####

#################################################


def mlmc_milstein(l, dt, N, M, coefs, avg_fun):
    """Performs one level of MLMC for the Milstein no area scheme

    Args:
        dt : time step
        N : number of time steps
        M : number of sample paths
        coefs : parameters of SDE
        avg_fun : evaulation function

    Returns:
        : the MLMC estimate, MLMC variance
    """
    # Generate samples of brownian motion
    bm_f = sqrt(dt) * torch.randn((M, N, 2), dtype=torch.float, device=device)
    # Coarse samples if necessary
    if l != 0:
        bm_c = combine_increments(bm_f, num_combines=1).cpu().numpy()
        dw1_c, dw2_c = bm_c[:, :, 0], bm_c[:, :, 1]

    bm_f = bm_f.cpu().numpy()
    # Extract increments
    dw1_f, dw2_f = bm_f[:, :, 0], bm_f[:, :, 1]

    # Initial conditions
    u0, v0 = coefs["u0"] * np.ones(shape=(M, 1)), coefs["v0"] * np.ones(shape=(M, 1))

    # Initialise all paths
    u_c, v_c, u_f, v_f = u0, v0, u0, v0

    # One step simulation on coarsest (zero) level
    if l == 0:
        for i in range(N):
            # Increment u and v
            [u_c, v_c] = milstein(dw1_f[:, [i]], dw2_f[:, [i]], dt, u_c, v_c, coefs)
        # Estimate and variance
        mc_estimator = avg_fun(u_c, v_c)
        mc_estimate = np.mean(mc_estimator)
        mc_var = np.var(mc_estimator)
        return mc_estimate, mc_var
    # Number of steps in coarse samples
    N_c = int(N / 2)
    dt_c = dt * 2
    for i in range(0, N_c):
        # increment u and v on coarser timescale
        [u_c, v_c] = milstein(dw1_c[:, [i]], dw2_c[:, [i]], dt_c, u_c, v_c, coefs)
        # For every coarse step, do 2 fine steps
        for j in range(0, 2):
            # increment u and v on finer timescale
            [u_f, v_f] = milstein(
                dw1_f[:, [2 * i + j]], dw2_f[:, [2 * i + j]], dt, u_f, v_f, coefs
            )

    # Estimate and variance
    mlmc_estimator = avg_fun(u_f, v_f) - avg_fun(u_c, v_c)
    mlmc_estimate = np.mean(mlmc_estimator)
    mlmc_variance = np.var(mlmc_estimator)

    return mlmc_estimate, mlmc_variance


def mlmc_milstein_anti(l, dt, N, M, coefs, avg_fun):
    """Performs one level of MLMC for the Milstein antithetic no area scheme

    Args:
        dt : time step
        N : number of time steps
        M : number of sample paths
        coefs : parameters of SDE
        avg_fun : evaulation function

    Returns:
        : the MLMC estimate, MLMC variance
    """
    # Generate samples of brownian motion
    bm_f = sqrt(dt) * torch.randn((M, N, 2), dtype=torch.float, device=device)
    # Coarse samples if necessary
    if l != 0:
        bm_c = combine_increments(bm_f, num_combines=1).cpu().numpy()
        dw1_c, dw2_c = bm_c[:, :, 0], bm_c[:, :, 1]

    bm_f = bm_f.cpu().numpy()
    # Extract increments
    dw1_f, dw2_f = bm_f[:, :, 0], bm_f[:, :, 1]

    # Initial conditions
    u0, v0 = coefs["u0"] * np.ones(shape=(M, 1)), coefs["v0"] * np.ones(shape=(M, 1))

    # Initialise all paths
    u_c, v_c, u_f, v_f, u_a, v_a = u0, v0, u0, v0, u0, v0

    # One step simulation on coarsest (zero) level
    if l == 0:
        for i in range(N):
            # Increment u and v
            [u_c, v_c] = milstein(dw1_f[:, [i]], dw2_f[:, [i]], dt, u_c, v_c, coefs)
        # Estimate and variance
        mc_estimator = avg_fun(u_c, v_c)
        mc_estimate = np.mean(mc_estimator)
        mc_var = np.var(mc_estimator)
        return mc_estimate, mc_var
    # Number of steps in coarse samples
    N_c = int(N / 2)
    dt_c = dt * 2
    for i in range(0, N_c):
        # increment u and v on coarser timescale
        [u_c, v_c] = milstein(dw1_c[:, [i]], dw2_c[:, [i]], dt_c, u_c, v_c, coefs)
        # For every coarse step, do 2 fine steps
        for j in range(0, 2):
            # increment u and v on finer timescale
            [u_f, v_f] = milstein(
                dw1_f[:, [2 * i + j]], dw2_f[:, [2 * i + j]], dt, u_f, v_f, coefs
            )
            [u_a, v_a] = milstein(
                dw1_f[:, [2 * i + 1 - j]],
                dw2_f[:, [2 * i + 1 - j]],
                dt,
                u_a,
                v_a,
                coefs,
            )

    # Estimate and variance
    mlmc_estimator = (1 / 2) * (avg_fun(u_f, v_f) + avg_fun(u_a, v_a)) - avg_fun(
        u_c, v_c
    )
    mlmc_estimate = np.mean(mlmc_estimator)
    mlmc_variance = np.var(mlmc_estimator)

    return mlmc_estimate, mlmc_variance


def mlmc_strang(l, dt, N, M, coefs, avg_fun, net=None, method=None):
    """Performs one level of MLMC for the Strang log-ODE scheme

    Args:
        dt : time step
        N : number of time steps
        M : number of sample paths
        coefs : parameters of SDE
        avg_fun : evaulation function

    Returns:
        _type_: the MLMC estimate, MLMC variance
    """
    # Generate increments and levy areas
    # samples_f, samples_c = net.generate_MC_samples(M, N, dt).cpu().numpy()
    if net is not None:
        samples_f = net.generate_MC_samples(M, N, dt)
    elif method == "Foster":
        samples_f = mlmc_foster(M, N, dt)
    elif method == "Rad":
        samples_f = mlmc_rad(M, N, dt)
    elif method == "NA":
        samples_f = sqrt(dt) * torch.randn((M, N, 3), dtype=torch.float, device=device)
        samples_f[:, :, 2] = samples_f[:, :, 2] * 0
    if l != 0:
        samples_c = MC_chen_combine(samples_f, num_combines=1).cpu().numpy()
        dw1_c = samples_c[:, :, 0]
        dw2_c = samples_c[:, :, 1]
        a12_c = samples_c[:, :, 2]

        if method == "NA":
            a12_c = a12_c * 0

    samples_f = samples_f.cpu().numpy()

    # Extract increments and areas
    dw1_f = samples_f[:, :, 0]
    dw2_f = samples_f[:, :, 1]
    a12_f = samples_f[:, :, 2]

    # Initial conditions
    u0, v0 = coefs["u0"] * np.ones(shape=(M, 1)), coefs["v0"] * np.ones(shape=(M, 1))

    # Initialise paths
    u_c, v_c, u_f, v_f = u0, v0, u0, v0

    # One step simulation on coarsest (zero) level
    if l == 0:
        for i in range(N):
            # Increment u and v
            [u_c, v_c] = strang_splitting(
                dw1_f[:, [i]], dw2_f[:, [i]], a12_f[:, [i]], dt, u_c, v_c, coefs
            )
        # Estimate and variance
        mc_estimator = avg_fun(u_c, v_c)
        mc_estimate = np.mean(mc_estimator)
        mc_var = np.var(mc_estimator)
        return mc_estimate, mc_var

    # Number of timesteps in coarse path
    N_c = int(N / 2)
    dt_c = dt * 2
    for i in range(0, N_c):
        # Increment u and v on coarser timescale
        [u_c, v_c] = strang_splitting(
            dw1_c[:, [i]], dw2_c[:, [i]], a12_c[:, [i]], dt_c, u_c, v_c, coefs
        )
        # for each xc step we do 2 xf steps
        for j in range(0, 2):
            # increment u and v on finer timescale
            [u_f, v_f] = strang_splitting(
                dw1_f[:, [2 * i + j]],
                dw2_f[:, [2 * i + j]],
                a12_f[:, [2 * i + j]],
                dt,
                u_f,
                v_f,
                coefs,
            )

    # Estimate and variance
    mlmc_estimator = avg_fun(u_f, v_f) - avg_fun(u_c, v_c)
    mlmc_estimate = np.mean(mlmc_estimator)
    mlmc_variance = np.var(mlmc_estimator)

    return mlmc_estimate, mlmc_variance


def mlmc_strang_anti(l, dt, N, M, coefs, avg_fun, net=None, method=None):
    """Performs one level of MLMC for the Strang log-ODE antithetic scheme

    Args:
        dt : time step
        N : number of time steps
        M : number of sample paths
        coefs : parameters of SDE
        avg_fun : evaulation function

    Returns:
        _type_: the MLMC estimate, MLMC variance
    """
    # Generate increments and levy areas
    # samples_f, samples_c = net.generate_MC_samples(M, N, dt).cpu().numpy()
    if net is not None:
        samples_f = net.generate_MC_samples(M, N, dt)
    elif method == "Foster":
        samples_f = mlmc_foster(M, N, dt)
    elif method == "Rad":
        samples_f = mlmc_rad(M, N, dt)
    elif method == "NA":
        samples_f = sqrt(dt) * torch.randn((M, N, 3), dtype=torch.float, device=device)
        samples_f[:, :, 2] = samples_f[:, :, 2] * 0
    if l != 0:
        samples_c = MC_chen_combine(samples_f, num_combines=1).cpu().numpy()
        dw1_c = samples_c[:, :, 0]
        dw2_c = samples_c[:, :, 1]
        a12_c = samples_c[:, :, 2]

        if method == "NA":
            a12_c = a12_c * 0

    samples_f = samples_f.cpu().numpy()

    # Extract increments and areas
    dw1_f = samples_f[:, :, 0]
    dw2_f = samples_f[:, :, 1]
    a12_f = samples_f[:, :, 2]

    # Initial conditions
    u0, v0 = coefs["u0"] * np.ones(shape=(M, 1)), coefs["v0"] * np.ones(shape=(M, 1))

    # Initialise paths
    u_c, v_c, u_f, v_f, u_a, v_a = u0, v0, u0, v0, u0, v0

    # One step simulation on coarsest (zero) level
    if l == 0:
        for i in range(N):
            # Increment u and v
            [u_c, v_c] = strang_splitting(
                dw1_f[:, [i]], dw2_f[:, [i]], a12_f[:, [i]], dt, u_c, v_c, coefs
            )
        # Estimate and variance
        mc_estimator = avg_fun(u_c, v_c)
        mc_estimate = np.mean(mc_estimator)
        mc_var = np.var(mc_estimator)
        return mc_estimate, mc_var

    # Number of timesteps in coarse path
    N_c = int(N / 2)
    dt_c = dt * 2
    for i in range(0, N_c):
        # Increment u and v on coarser timescale
        [u_c, v_c] = strang_splitting(
            dw1_c[:, [i]], dw2_c[:, [i]], a12_c[:, [i]], dt_c, u_c, v_c, coefs
        )
        # for each xc step we do 2 xf steps
        for j in range(0, 2):
            # increment u and v on finer timescale
            [u_f, v_f] = strang_splitting(
                dw1_f[:, [2 * i + j]],
                dw2_f[:, [2 * i + j]],
                a12_f[:, [2 * i + j]],
                dt,
                u_f,
                v_f,
                coefs,
            )
            [u_a, v_a] = strang_splitting(
                dw1_f[:, [2 * i + 1 - j]],
                dw2_f[:, [2 * i + 1 - j]],
                a12_f[:, [2 * i + 1 - j]],
                dt,
                u_a,
                v_a,
                coefs,
            )

    # Estimate and variance
    mlmc_estimator = (1 / 2) * (avg_fun(u_f, v_f) + avg_fun(u_a, v_a)) - avg_fun(
        u_c, v_c
    )
    mlmc_estimate = np.mean(mlmc_estimator)
    mlmc_variance = np.var(mlmc_estimator)

    return mlmc_estimate, mlmc_variance


#################################################

#### Full MLMC Scheme ####

#################################################


def do_mlmc(coefs, avg_fun, M_list, dt_list, scheme="Milstein", net=None):
    """Performs MLMC for given number of levels, samples and timesteps.

    Args:
        M_list (_type_): Number of samples on each level.
        dt_list (_type_): Timestep on each level.
        scheme (str, optional): Numerical scheme. Defaults to "Milstein".
    """
    # Initialise outputs
    level_estimates = []
    level_variances = []
    T = coefs["T"]
    # Loop over levels
    for l in range(len(M_list)):
        # Initialise variables
        M = M_list[l]
        dt = dt_list[l]
        N = int(T / dt_list[l])
        if scheme == "Milstein":
            mlmc_estimate, mlmc_variance = mlmc_milstein(l, dt, N, M, coefs, avg_fun)
        elif scheme == "Strang":
            mlmc_estimate, mlmc_variance = mlmc_strang(
                l, dt, N, M, coefs, avg_fun, net=net
            )
        elif scheme == "Milstein_anti":
            mlmc_estimate, mlmc_variance = mlmc_milstein_anti(
                l, dt, N, M, coefs, avg_fun
            )
        elif scheme == "Strang_anti":
            mlmc_estimate, mlmc_variance = mlmc_strang_anti(
                l, dt, N, M, coefs, avg_fun, net=net
            )
        elif scheme == "Strang_F":
            mlmc_estimate, mlmc_variance = mlmc_strang(
                l, dt, N, M, coefs, avg_fun, method="Foster"
            )
        elif scheme == "Strang_R":
            mlmc_estimate, mlmc_variance = mlmc_strang(
                l, dt, N, M, coefs, avg_fun, method="Rad"
            )
        elif scheme == "Strang_NA":
            mlmc_estimate, mlmc_variance = mlmc_strang(
                l, dt, N, M, coefs, avg_fun, method="NA"
            )
        elif scheme == "Strang_NA_anti":
            mlmc_estimate, mlmc_variance = mlmc_strang_anti(
                l, dt, N, M, coefs, avg_fun, method="NA"
            )
        # Store results
        level_estimates.append(mlmc_estimate)
        level_variances.append(mlmc_variance)

    # Calculate total estimate and variance
    total_estimate = np.sum(level_estimates)
    total_variance = np.sum(level_variances)

    return total_estimate, total_variance, level_estimates, level_variances


#################################################

#### Optimal MLMC Parameters ####

#################################################


def do_mlmc_giles(coefs, avg_fun, dt0=1, scheme="Milstein", accuracy=0.1, net=None):
    """Implements the Giles' MLMC algorithm for given number of levels, samples and timesteps.

    Args:
        coefs (_type_): coefficients of SDE
        avg_fun (_type_): payoff function
        dt0 (int, optional): initial timestep. Defaults to 1.
        scheme (str, optional): Milstein or Strang. Defaults to "Milstein".
        accuracy (float, optional): Desired RMSE. Defaults to 0.1.
        net (_type_, optional): If using Strang, the generator for Levy area samples. Defaults to None.

    Returns:
        float: MLMC estimate
    """

    # Initial number of levels and samples
    L = -1
    M = 10**4
    N0 = int(1 / dt0)
    T = coefs["T"]
    Var_list = []
    dt_list = []
    MC_list = []

    # Determine variance and number of samples on coarsest two levels, and generate aditional samples if needed
    for j in range(2):
        # Updated L
        L += 1
        N = N0 * int(2**L)
        dt = T / N

        # Estimate variance on level L
        if scheme == "Milstein":
            MC_L, V_L = mlmc_milstein(L, dt, N, M, coefs, avg_fun)
        elif scheme == "Strang":
            MC_L, V_L = mlmc_strang(L, dt, N, M, coefs, avg_fun, net=net)

        # Estimate number of samples needed on level L
        Var_list.append(V_L)
        dt_list.append(dt)
        M_L = int(
            np.ceil(
                2
                * np.sqrt(V_L * dt)
                * np.sum([np.sqrt(Var_list[j] / dt_list[j]) for j in range(L + 1)])
                / accuracy**2
            )
        )

        # Check if additional samples are needed
        if M_L > M:
            # Compute estimate with extra samples
            if scheme == "Milstein":
                MC_L = (
                    (M_L - M) * mlmc_milstein(L, dt, N, M_L - M, coefs, avg_fun)[0]
                    + MC_L * M
                ) / M_L
            elif scheme == "Strang":
                MC_L = (
                    (M_L - M)
                    * mlmc_strang(L, dt, N, M_L - M, coefs, avg_fun, net=net)[0]
                    + MC_L * M
                ) / M_L
        # Store estimate
        MC_list.append(MC_L)

    # Desired
    if scheme == "Milstein":
        desired_bias = (1 / np.sqrt(2)) * (2 - 1) * accuracy
    elif scheme == "Strang":
        desired_bias = (1 / np.sqrt(2)) * (2**2 - 1) * accuracy
    current_bias = desired_bias + 1

    # Loop until bias is less than desired bias
    while current_bias > desired_bias:
        # Update L
        L += 1
        N = N0 * int(2**L)
        dt = T / N
        # Estimate variance on level L
        if scheme == "Milstein":
            MC_L, V_L = mlmc_milstein(L, dt, N, M, coefs, avg_fun)
        elif scheme == "Strang":
            MC_L, V_L = mlmc_strang(L, dt, N, M, coefs, avg_fun, net=net)
        # Estimate number of samples needed on level L
        Var_list.append(V_L)
        dt_list.append(dt)
        # Number of samples needed
        M_L = int(
            np.ceil(
                2
                * np.sqrt(V_L * dt)
                * np.sum([np.sqrt(Var_list[j] / dt_list[j]) for j in range(L + 1)])
                / accuracy**2
            )
        )
        # Check if additional samples are needed
        if M_L > M:
            # Compute estimate with extra samples
            if scheme == "Milstein":
                MC_L = (
                    (M_L - M) * mlmc_milstein(L, dt, N, M_L - M, coefs, avg_fun)[0]
                    + MC_L * M
                ) / M_L
            elif scheme == "Strang":
                MC_L = (
                    (M_L - M)
                    * mlmc_strang(L, dt, N, M_L - M, coefs, avg_fun, net=net)[0]
                    + MC_L * M
                ) / M_L
        # Store estimate
        MC_list.append(MC_L)

        # Update Current Bias
        current_bias = np.abs(MC_list[-1])

    # Calculate MLMC estimate
    mlmc_estimate = np.sum(MC_list)
    var_estimate = np.sum(Var_list)

    return mlmc_estimate, var_estimate


def levy_bias_estimator(coefs, avg_fun, M, dt=0.1, net=None):
    """Computes the bias introduced by fake levy area samples.

    Args:
        coefs (_type_): coefficients of SDE
        avg_fun (_type_): payoff function
        M (_type_): Number of samples
        dt (float, optional): timestep. Defaults to 0.1.
        net (_type_, optional): generator for levy area samples. Defaults to None.
    """
    T = coefs["T"]
    N = int(T / dt)
    # Generate samples
    samples_coarse = net.generate_MC_samples(M, N, dt).cpu().numpy()
    # samples_fine = net.generate_MC_samples(M, N, dt).cpu().numpy()
    samples_fine = net.generate_MC_samples(M, 2 * N, dt / 2)
    samples_fine = MC_chen_combine(samples_fine, num_combines=1).cpu().numpy()

    # Scheme using coarse samples
    u_c, v_c = coefs["u0"] * np.ones(shape=(M, 1)), coefs["v0"] * np.ones(shape=(M, 1))
    for i in range(0, N):
        [u_c, v_c] = strang_splitting(
            samples_coarse[:, [i], 0],
            samples_coarse[:, [i], 1],
            samples_coarse[:, [i], 2],
            dt,
            u_c,
            v_c,
            coefs,
        )
    # Coarse Estimate
    coarse_estimator = avg_fun(u_c, v_c)
    coarse_estimate = np.mean(coarse_estimator)

    # Scheme using fine samples
    u_f, v_f = coefs["u0"] * np.ones(shape=(M, 1)), coefs["v0"] * np.ones(shape=(M, 1))
    for i in range(0, N):
        [u_f, v_f] = strang_splitting(
            samples_fine[:, [i], 0],
            samples_fine[:, [i], 1],
            samples_fine[:, [i], 2],
            dt,
            u_f,
            v_f,
            coefs,
        )
    # Fine Estimate
    fine_estimator = avg_fun(u_f, v_f)
    fine_estimate = np.mean(fine_estimator)
    # Bias
    bias = fine_estimate - coarse_estimate
    return bias
