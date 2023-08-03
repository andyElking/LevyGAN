# Import packages

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import torch
import torch.nn as nn
from src.model.Generator import PairNetGenerator
from src.aux_functions import MC_chen_combine, mom4_gpu, Davie_gpu_all, Davie_mom2, rademacher_GPU_dim2, rademacher_GPU_dim2_var
from math import sqrt

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


"""We compare various MLMC methods: Milstein, Milstein Antithetic, and Strang log-ODE with three different fake levy areas
"""

# Plot colorus
colors = ["#60ceaf",
"#de76bc",
"#81cb61",
"#f90031",
"#8449be",
"#5ca9ff"]


#####################################

    ## Load Generator ## 

#####################################

gen_config = gen_config = {
            'use_pair_net': True,
            'bm_dim': 2,
            'noise_size': 4,  # size of latent space
            'num_layers': 3,
            'hidden_dim': 16,
            'activation': ('leaky', 0.01),  # or 'relu'
            'batch_norm': True,
            'pairnet_bn': True,
            'do_bridge_flipping': True,  # "bridge_flipping", otherwise off
            'use_mixed_noise': False,  # Uses noise from several distributions. Gives bad results for now...
            'use_attention': False,
            'enforce_antisym': False,
            'reinject_h': False,
            'gen_dict_saving_on': True,
                }

generator = PairNetGenerator(gen_config)
generator.load_dict(filename="./good_model_saves/generator_4d_PairNet3LAY_16HID_lky0.01lky0.01ACT_BN_4noise_bf/gen_num5_best__scr.pt")
device = 'cuda'


############################################

    ## One Step of Numerical Schemes ## 

############################################


def milstein_na(dw1,dw2,dt,u,v,coefs):
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
    u_new = u + (coefs["r"] - 0.5 * v)*dt + np.sqrt(np.maximum(0,v))*dw1 + 0.25*coefs["sigma"]*dw1*dw2
    v_new = v + coefs["k"]*(coefs["theta"] - v)*dt + coefs["sigma"]*np.sqrt(np.maximum(0,v))*dw2 + 0.25*coefs["sigma"]**2 * (dw2**2 - dt)

    return u_new,v_new

def strang_splitting(dw1,dw2,a12,dt,u,v,coefs):
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
    xi = coefs["theta"] - coefs["sigma"]**2 / (4*coefs["k"])
    sig = coefs["sigma"]; kappa = coefs["k"]; r = coefs["r"]
    
    # solving drift to half timestep
    v_d = xi + (v-xi)*np.exp(-0.5*kappa*dt)
    u_d = u + 0.5*dt*(r-0.5*xi) + 0.5*(1/kappa)*(v-xi)*(np.exp(-0.5*kappa*dt)-1)
    # solving diffusion
    v_dd = (np.sqrt(v_d) + 0.5*sig*dw2)**2
    u_dd = u_d + np.sqrt(v_d)*dw1 + 0.25*sig*dw1*dw2 - 0.5*sig*a12
    # solving drift 2nd half timestep
    v_new = xi + (v_dd-xi)*np.exp(-0.5*kappa*dt)
    u_new = u_dd + 0.5*dt*(r-0.5*xi) + 0.5*(1/kappa)*(v_dd-xi)*(np.exp(-0.5*kappa*dt)-1)
    
    return u_new,v_new


#################################################

    ## MLMC Schemes for One Level difference ## 

#################################################


def mlmc_milstein_na(T,N,M,u0_const,v0_const,coefs,avg_fun):
    """Performs one level of MLMC for the Milstein no area  scheme

    Args:
        T : terminal time
        N : number of time steps
        M : number of sample paths
        u0_const : initial u
        v0_const : initial v
        coefs : parameters of SDE
        avg_fun : evaulation function

    Returns:
        : the MLMC estimate, MC estimate, and variance
    """
    dt = T/N
    dw1 = np.random.normal(size=(M,N),scale=np.sqrt(dt))
    dw2 = np.random.normal(size=(M,N),scale=np.sqrt(dt))
    
    u0 = u0_const*np.ones(shape=(M,1))
    v0 = v0_const*np.ones(shape=(M,1))

    # for each time step, we compute xc, xf (for u and v)
    u_c = u0; v_c = v0; u_f = u0; v_f = v0; u_a = u0; v_a = v0  # initialising all paths
    

    if N == 1:  # one step simulation on coarsest (zero) level
        [u_c,v_c] = milstein_na(dw1,dw2,dt,u_c,v_c,coefs)
        mc_estimate = np.mean(avg_fun(u_c,v_c))
        return mc_estimate
    else:
        # idea: iterate the number of steps in xc
        N_c = int(N / 2)
        dt_c = dt*2
        for i in range(0,N_c):
            # sum up Brownian increments to coarser timescale
            dw1_curr = dw1[:,2*i:2*(i+1)]
            dw2_curr = dw2[:,2*i:2*(i+1)]
            dw1_c = np.sum(dw1_curr,axis=1,keepdims=True)
            dw2_c = np.sum(dw2_curr,axis=1,keepdims=True)
            # increment u and v on coarser timescale
            [u_c,v_c] = milstein_na(dw1_c,dw2_c,dt_c,u_c,v_c,coefs)
            # for each xc step we do 2 xf steps
            for j in range(0,2):
                # save correct (finer) Brownian increments
                dw1_f = dw1_curr[:,[j]]
                dw2_f = dw2_curr[:,[j]]
                # increment u and v on finer timescale
                [u_f,v_f] = milstein_na(dw1_f,dw2_f,dt,u_f,v_f,coefs)
    mlmc_estimator = avg_fun(u_f,v_f) - avg_fun(u_c,v_c)

                
    mlmc_estimate = np.mean(mlmc_estimator)
    mlmc_variance = np.var(mlmc_estimator)
    mc_estimate = np.mean(avg_fun(u_f,v_f))
    return mlmc_estimate, mc_estimate, mlmc_variance

def mlmc_antithetic_na(T,N,M,u0_const,v0_const,coefs,avg_fun):
    """Performs one level of MLMC for the Milstein-antithetic no area scheme

    Args:
        T : terminal time
        N : number of time steps
        M : number of sample paths
        u0_const : initial u
        v0_const : initial v
        coefs : parameters of SDE
        avg_fun : evaulation function

    Returns:
        : the MLMC estimate, MC estimate, and variance
    """
    dt = T/N
    dw1 = np.random.normal(size=(M,N),scale=np.sqrt(dt))
    dw2 = np.random.normal(size=(M,N),scale=np.sqrt(dt))
    
    u0 = u0_const*np.ones(shape=(M,1))
    v0 = v0_const*np.ones(shape=(M,1))

    # for each time step, we compute xc, xf, and xa (for u and v)
    u_c = u0; v_c = v0; u_f = u0; v_f = v0; u_a = u0; v_a = v0  # initialising all paths
    
    if N == 1:  # one step simulation on coarsest (zero) level
        [u_c,v_c] = milstein_na(dw1,dw2,dt,u_c,v_c,coefs)
        mc_estimate = np.mean(avg_fun(u_c,v_c))
        return mc_estimate
    else:
        # idea: iterate the number of steps in xc
        N_c = int(N / 2)
        dt_c = dt*2
        for i in range(0,N_c):
            # sum up Brownian increments to coarser timescale
            dw1_curr = dw1[:,2*i:2*(i+1)]
            dw2_curr = dw2[:,2*i:2*(i+1)]
            dw1_c = np.sum(dw1_curr,axis=1,keepdims=True)
            dw2_c = np.sum(dw2_curr,axis=1,keepdims=True)
            # increment u and v on coarser timescale
            [u_c,v_c] = milstein_na(dw1_c,dw2_c,dt_c,u_c,v_c,coefs)
            # for each xc step we do 2 xf (and xa) steps
            for j in range(0,2):
                # save correct (finer) Brownian increments
                dw1_f = dw1_curr[:,[j]]
                dw2_f = dw2_curr[:,[j]]
                dw1_a = dw1_curr[:,[1-j]]
                dw2_a = dw2_curr[:,[1-j]]
                # increment u and v on finer (and antithetic) timescale
                [u_f,v_f] = milstein_na(dw1_f,dw2_f,dt,u_f,v_f,coefs)
                [u_a,v_a] = milstein_na(dw1_a,dw2_a,dt,u_a,v_a,coefs)
    antithetic_estimator = 0.5*(avg_fun(u_f,v_f) + avg_fun(u_a,v_a)) - avg_fun(u_c,v_c)           
    mlmc_a_estimate = np.mean(antithetic_estimator)
    mlmc_a_variance = np.var(antithetic_estimator)
    mc_estimate = np.mean(avg_fun(u_f,v_f))
    mlmc_estimate = np.mean(avg_fun(u_f,v_f) - avg_fun(u_c,v_c))
    return mlmc_a_estimate,mlmc_estimate,mc_estimate,mlmc_a_variance

def mlmc_strang(T,N,M,u0_const,v0_const,coefs,avg_fun, gen = "foster"):
    """Performs one level of MLMC for the Strang log-ODE scheme

    Args:
        T : terminal time
        N : number of time steps
        M : number of sample paths
        u0_const : initial u
        v0_const : initial v
        coefs : parameters of SDE
        avg_fun : evaulation function
        gen (str, optional): type of Levy area to use. Defaults to "foster".

    Returns:
        _type_: the MLMC estimate, MC estimate, and variance
    """
    dt = T/N
    # Generate samples of brownian motion and levy area either using Foster or PairNet, or Rademacher
    if gen == "foster":
        samples = mom4_gpu(2, M*N)
        samples[:, :2] = samples[:, :2]*sqrt(dt)
        samples[:, 2:] = samples[:, 2:]*dt
        samples = samples.view(M, N, -1)
    elif gen == "net":
        samples = generator.generate_MC_samples(M, N, dt) 
    elif gen == "rademacher":
        samples = rademacher_GPU_dim2_var(M*N)
        samples[:, :2] = samples[:, :2]*sqrt(dt)
        samples[:, 2:] = samples[:, 2:]*dt
        samples = samples.view(M, N, -1)
    if N >  1:
        samples_c = MC_chen_combine(samples, 2).cpu().numpy()
    samples = samples.cpu().numpy()
    dw1_f = samples[:, :, 0]
    dw2_f = samples[:, :, 1]
    a12_f = samples[:, :, 2]
    if N > 1:
        dw1_c = samples_c[:, :, 0]
        dw2_c = samples_c[:, :, 1]
        a12_c = samples_c[:, :, 2]
    
    u0 = u0_const*np.ones(shape=(M,1))
    v0 = v0_const*np.ones(shape=(M,1))

    # for each time step, we compute xc, xf, and xa (for u and v)
    u_c = u0; v_c = v0; u_f = u0; v_f = v0  # initialising all paths

    if N == 1:  # one step simulation on coarsest (zero) level
        [u_c,v_c] = strang_splitting(dw1_f,dw2_f,a12_f,dt,u_c,v_c,coefs)
        mc_estimate = np.mean(avg_fun(u_c,v_c))
        return mc_estimate
    else:
        # iterate the number of steps in xc
        N_c = int(N / 2)
        dt_c = dt*2
        for i in range(0,N_c):
            # sum up Brownian increments to coarser timescale
            dw1_curr = dw1_f[:,2*i:2*(i+1)]
            dw2_curr = dw2_f[:,2*i:2*(i+1)]
            a12_curr = a12_f[:,2*i:2*(i+1)]
            # Get Levy area on coarse scale
            dw1_c_ = dw1_c[:, [i]]
            dw2_c_ = dw2_c[:, [i]]
            a12_c_ = a12_c[:, [i]]
            # increment u and v on coarser timescale
            [u_c,v_c] = strang_splitting(dw1_c_,dw2_c_,a12_c_,dt_c,u_c,v_c,coefs)
            # for each xc step we do 2 xf steps
            for j in range(0,2):
                # save correct (finer) Brownian increments
                dw1_f_ = dw1_curr[:,[j]]
                dw2_f_ = dw2_curr[:,[j]]
                # get levy area on finer timescale
                a12_f_ = a12_curr[:,[j]]
                # increment u and v on finer timescale
                [u_f,v_f] = strang_splitting(dw1_f_,dw2_f_,a12_f_,dt,u_f,v_f,coefs)
    mlmc_estimator = avg_fun(u_f,v_f) - avg_fun(u_c,v_c)
               
    mlmc_estimate = np.mean(mlmc_estimator)
    mlmc_variance = np.var(mlmc_estimator)
    mc_estimate = np.mean(avg_fun(u_f,v_f))
    return mlmc_estimate, mc_estimate, mlmc_variance

def mlmc_strang_na(T,N,M,u0_const,v0_const,coefs,avg_fun):
    """Performs one level of MLMC for the Strang splitting scheme (No Area)

    Args:
        T : terminal time
        N : number of time steps
        M : number of sample paths
        u0_const : initial u
        v0_const : initial v
        coefs : parameters of SDE
        avg_fun : evaulation function
        gen (str, optional): type of Levy area to use. Defaults to "foster".

    Returns:
        _type_: the MLMC estimate, MC estimate, and variance
    """
    dt = T/N
    # Generate samples of brownian motion and levy area either using Foster or PairNet
    dw1 = np.random.normal(size=(M,N),scale=np.sqrt(dt))
    dw2 = np.random.normal(size=(M,N),scale=np.sqrt(dt))
    
    u0 = u0_const*np.ones(shape=(M,1))
    v0 = v0_const*np.ones(shape=(M,1))

    # for each time step, we compute xc, xf, and xa (for u and v)
    u_c = u0; v_c = v0; u_f = u0; v_f = v0  # initialising all paths

    if N == 1:  # one step simulation on coarsest (zero) level
        [u_c,v_c] = strang_splitting(dw1,dw2,0,dt,u_c,v_c,coefs)
        mc_estimate = np.mean(avg_fun(u_c,v_c))
        return mc_estimate
    else:
        # idea: iterate the number of steps in xc
        N_c = int(N / 2)
        dt_c = dt*2
        for i in range(0,N_c):
            # sum up Brownian increments to coarser timescale
            dw1_curr = dw1[:,2*i:2*(i+1)]
            dw2_curr = dw2[:,2*i:2*(i+1)]
            dw1_c = np.sum(dw1_curr,axis=1,keepdims=True)
            dw2_c = np.sum(dw2_curr,axis=1,keepdims=True)
            # increment u and v on coarser timescale
            [u_c,v_c] = strang_splitting(dw1_c,dw2_c,0,dt_c,u_c,v_c,coefs)
            # for each xc step we do 2 xf steps
            for j in range(0,2):
                # save correct (finer) Brownian increments
                dw1_f = dw1_curr[:,[j]]
                dw2_f = dw2_curr[:,[j]]
                # increment u and v on finer timescale
                [u_f,v_f] = strang_splitting(dw1_f,dw2_f,0,dt,u_f,v_f,coefs)
    mlmc_estimator = avg_fun(u_f,v_f) - avg_fun(u_c,v_c)
         
    mlmc_estimate = np.mean(mlmc_estimator)
    mlmc_variance = np.var(mlmc_estimator)
    mc_estimate = np.mean(avg_fun(u_f,v_f))
    return mlmc_estimate, mc_estimate, mlmc_variance


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
    alpha = -w*w/2 - 1j*w/2
    beta = a - rho*vvol*1j*w
    gamma = vvol*vvol/2
    h = np.sqrt(beta*beta - 4*alpha*gamma)
    rplus = (beta + h)/vvol/vvol
    rminus = (beta - h)/vvol/vvol
    g=rminus/rplus
    # Required inputs for the characteristic function
    C = a * (rminus * t - (2 / vvol**2) * np.log((1 - g * np.exp(-h*t))/(1-g)))
    D = rminus * (1 - np.exp(-h * t))/(1 - g * np.exp(-h*t))
    # Characteristic function evaluated at points w
    y = np.exp(C*vbar + D*v0 + 1j*w*np.log(s0*np.exp(r*t)))
    return y

# this still needs "translated" - look at numerical integration and inline functions!
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
    int1 = lambda w, s0, v0, vbar, a, vvol, r, rho, t, k : np.real(np.exp(-1j*w*np.log(k))*chfun_heston(s0, v0, vbar, a, vvol, r,
    rho, t, w-1j)/(1j*w*chfun_heston(s0, v0, vbar, a, vvol, r, rho, t, -1j))) # inner integral1
    int1_val,tol1 = integrate.quad(lambda w : int1(w,s0, v0, vbar, a, vvol, r, rho, t, k),0,100); # numerical integration
    pi1 = int1_val/np.pi+0.5 # final pi1
     # Inner integral 2:
    int2 = lambda w, s0, v0, vbar, a, vvol, r, rho, t, k : np.real(np.exp(-1j*w*np.log(k))*chfun_heston(s0, v0, vbar, a, vvol, r,
    rho, t, w)/(1j*w))
    int2_val,tol2 = integrate.quad(lambda w :int2(w,s0, v0, vbar, a, vvol, r, rho, t, k),0,100);int2_val = np.real(int2_val)
    pi2 = int2_val/np.pi+0.5 # final pi2
    # 3rd step: calculate call value
    y = s0*pi1-np.exp(-r*t)*k*pi2

    return y
##################################################

            ##### Testing #####
            
##################################################

# Parameters of SDE and payoff function
np.random.seed(0)
coefs = {"r":0.1, "k":2, "theta":0.1, "sigma":0.6} 
u0 = np.log(20); v0 = 2 # initial condition for u and v
R = coefs["r"]; K = 20; T = 1
payoff = lambda x,y: np.exp(-R*T)*np.maximum(0,(np.exp(x)-K))
# Exact value of call option
exact_val = call_heston_cf(s0=np.exp(u0), v0=v0, vbar=coefs["theta"], a=coefs["k"], vvol=coefs["sigma"], r=R, rho=0, t=T, k=K)

##################################################

        ##### Compare Area vs No Area #####
            
##################################################

# Define number of levels and number of times to run experiments
L = 5; num_iters = 24
out_array_variances = np.zeros((3, num_iters, L))
out_array_errors = np.zeros((3, num_iters, L))

# Run experiment num_iters times
for iteration in range(num_iters):
    # MLMC parameters
    M0 = 2**27
    N = 1; M = M0
    # Initialise the level 0 MLMC
    # STRANG G: Strang log-ODE with PairNet Levy area
    strangG_mlmc_sum = mlmc_strang(T,N,M,u0,v0,coefs,payoff, gen = "net")  # level 0 term
    strangG_ml_var = np.zeros(shape=(L))
    strangG_mlmc_hist = np.zeros(shape=(L))
    strangG_mlmc_hist[0] = strangG_mlmc_sum
    # MILSTEIN ANTITHETIC NA
    milstein_anti_mlmc_sum = mlmc_antithetic_na(T,N,M,u0,v0,coefs,payoff)
    milstein_anti_ml_var = np.zeros(shape=(L))
    milstein_anti_mlmc_hist = np.zeros(shape=(L))
    milstein_anti_mlmc_hist[0] = milstein_anti_mlmc_sum
    # MILSTEIN NA
    milstein_mlmc_sum = mlmc_milstein_na(T,N,M,u0,v0,coefs,payoff)
    milstein_ml_var = np.zeros(shape=(L))
    milstein_mlmc_hist = np.zeros(shape=(L))
    milstein_mlmc_hist[0] = milstein_mlmc_sum

    # Perform MLMC
    levels = np.zeros(shape=(L))
    timesteps = np.zeros(shape=(L))
    timesteps[0] = T/N
    # Loop over number of levels
    for l in range(1,L):
        N = int(2**l); M = int(M0 * 2**(-l))
        # STRANG G
        [strangG_mlmc_term, strangG_var_term] = mlmc_strang(T,N,M,u0,v0,coefs,payoff, gen = "net")[0::2]
        strangG_mlmc_sum = strangG_mlmc_sum + strangG_mlmc_term
        strangG_ml_var[l] = strangG_var_term
        # MILSTEIN ANTITHETIC NA
        [milstein_anti_mlmc_term, milstein_anti_var_term] = mlmc_antithetic_na(T,N,M,u0,v0,coefs,payoff)[0::3]
        milstein_anti_mlmc_sum = milstein_anti_mlmc_sum + milstein_anti_mlmc_term
        milstein_anti_ml_var[l] = milstein_anti_var_term
        # MILSTEIN NA
        [milstein_mlmc_term, milstein_var_term] = mlmc_milstein_na(T,N,M,u0,v0,coefs,payoff)[0::2]
        milstein_mlmc_sum = milstein_mlmc_sum + milstein_mlmc_term
        milstein_ml_var[l] = milstein_var_term

        # Levels
        levels[l] = l
        timesteps[l] = T/N
        strangG_mlmc_hist[l] = strangG_mlmc_sum
        milstein_anti_mlmc_hist[l] = milstein_anti_mlmc_sum
        milstein_mlmc_hist[l] = milstein_mlmc_sum

    # Compute the error at each level
    strangG_error = np.abs(strangG_mlmc_hist - exact_val)
    milstein_anti_error = np.abs(milstein_anti_mlmc_hist - exact_val)
    milstein_error = np.abs(milstein_mlmc_hist - exact_val)
    # Record errors and variances for the iteration
    out_array_variances[0, iteration, :] = strangG_ml_var
    out_array_variances[1, iteration, :] = milstein_anti_ml_var
    out_array_variances[2, iteration, :] = milstein_ml_var
    out_array_errors[0, iteration, :] = strangG_error
    out_array_errors[1, iteration, :] = milstein_anti_error
    out_array_errors[2, iteration, :] = milstein_error


# PLOTTING
# Variance reduction by level
strangG_v_mean = np.log(out_array_variances[0, :, :].mean(axis = 0)[1:])
milstein_anti_v_mean = np.log(out_array_variances[1, :, :].mean(axis = 0)[1:])
milstein_v_mean = np.log(out_array_variances[2, :, :].mean(axis = 0)[1:])
plt.plot(levels[1::],np.log(120)+(-1.35*levels[1::]),linestyle='-.', color = 'black')
plt.plot(levels[1::],milstein_v_mean,marker='^',linewidth=2, color = colors[0],linestyle=':')
plt.plot(levels[1::],milstein_anti_v_mean,marker='v',linewidth=2,color = colors[1],linestyle='--')
plt.plot(levels[1::],strangG_v_mean,marker='s',linewidth=2, color = colors[2])
plt.title('Multilevel Variance')
plt.xlabel('Level')
plt.xticks(range(1,L))
plt.ylabel('log(var)')
plt.legend(['Gradient -1.35','Milstein', 'Milstein-anti','Strang G'])
plt.savefig('fake_vs_no_fake_variance_.png')
plt.close()

# plotting error vs level
strangG_e_mean = np.log(out_array_errors[0, :, :]).mean(axis = 0)
milstein_anti_e_mean = np.log(out_array_errors[1, :, :]).mean(axis = 0)
milstein_e_mean = np.log(out_array_errors[2, :, :]).mean(axis = 0)
strangG_e_std = np.log(out_array_errors[0, :, :]).std(axis = 0)
milstein_anti_e_std = np.log(out_array_errors[1, :, :]).std(axis = 0)
milstein_e_std = np.log(out_array_errors[2, :, :]).std(axis = 0)
# plt.yscale('log')
plt.plot(levels[0::],-1*levels[0::]+1,linestyle='-.', color = 'black')
plt.plot(levels[0::],milstein_e_mean,marker='^',linewidth=2, color = colors[0],linestyle=':')
plt.plot(levels[0::],milstein_anti_e_mean,marker='v',linewidth=2,color = colors[1],linestyle='--')
plt.plot(levels[0::],strangG_e_mean,marker='s',linewidth=2, color = colors[2])
plt.fill_between(levels[0::], milstein_e_mean-milstein_e_std, milstein_e_mean+milstein_e_std, alpha = 0.3, facecolor=colors[0])
plt.fill_between(levels[0::], milstein_anti_e_mean-milstein_anti_e_std, milstein_anti_e_mean+milstein_anti_e_std, alpha = 0.3, facecolor=colors[1])
plt.fill_between(levels[0::], strangG_e_mean-strangG_e_std, strangG_e_mean+strangG_e_std, alpha = 0.3, facecolor=colors[2])
plt.plot(levels[0::],-1.38*levels[0::],linestyle='-.',color='black')
plt.title('Empirical Error')
plt.xlabel('Level')
plt.xticks(range(0,L))
plt.ylabel('log(error)')
plt.legend(['Gradients -1 and -1.38','Milstein', 'Milstein-anti','Strang G'])
plt.savefig('fake_vs_no_fake_error_.png')
plt.close()

##################################################

            ##### Compare Areas #####
            
##################################################

# Define number of levels and number of times to run experiments
L = 5; num_iters = 24
out_array_variances = np.zeros((4, num_iters, L))
out_array_errors = np.zeros((4, num_iters, L))

# Run experiment num_iters number of times
for iteration in range(num_iters):
    # MLMC parameters
    M0 = int(2**27)
    N = 1; M = M0; 
    # Initialise the level 0 MLMC
    
    # STRANG F
    strangF_mlmc_sum = mlmc_strang(T,N,M,u0,v0,coefs,payoff, gen = "foster")  # level 0 term
    strangF_ml_var = np.zeros(shape=(L))
    strangF_mlmc_hist = np.zeros(shape=(L))
    strangF_mlmc_hist[0] = strangF_mlmc_sum
    # STRANG R
    strangR_mlmc_sum = mlmc_strang(T,N,M,u0,v0,coefs,payoff, gen = "rademacher")  # level 0 term
    strangR_ml_var = np.zeros(shape=(L))
    strangR_mlmc_hist = np.zeros(shape=(L))
    strangR_mlmc_hist[0] = strangR_mlmc_sum
    # STRANG G
    strangG_mlmc_sum = mlmc_strang(T,N,M,u0,v0,coefs,payoff, gen = "net")  # level 0 term
    strangG_ml_var = np.zeros(shape=(L))
    strangG_mlmc_hist = np.zeros(shape=(L))
    strangG_mlmc_hist[0] = strangG_mlmc_sum
    # STRANG NA
    strangNA_mlmc_sum = mlmc_strang_na(T,N,M,u0,v0,coefs,payoff)  # level 0 term
    strangNA_ml_var = np.zeros(shape=(L))
    strangNA_mlmc_hist = np.zeros(shape=(L))
    strangNA_mlmc_hist[0] = strangNA_mlmc_sum

    # Perform MLMC
    levels = np.zeros(shape=(L))
    timesteps = np.zeros(shape=(L,1))
    timesteps[0] = T/N
    # Loop over levels
    for l in range(1,L):
        N = int(2**l); M = int(M0 * 2**(-l))
        # STRANG
        [strangF_mlmc_term, strangF_var_term] = mlmc_strang(T,N,M,u0,v0,coefs,payoff, gen = "foster")[0::2]
        strangF_mlmc_sum = strangF_mlmc_sum + strangF_mlmc_term
        strangF_ml_var[l] = strangF_var_term
        [strangR_mlmc_term, strangR_var_term] = mlmc_strang(T,N,M,u0,v0,coefs,payoff, gen = "rademacher")[0::2]
        strangR_mlmc_sum = strangR_mlmc_sum + strangR_mlmc_term
        strangR_ml_var[l] = strangR_var_term
        [strangG_mlmc_term, strangG_var_term] = mlmc_strang(T,N,M,u0,v0,coefs,payoff, gen = "net")[0::2]
        strangG_mlmc_sum = strangG_mlmc_sum + strangG_mlmc_term
        strangG_ml_var[l] = strangG_var_term
        [strangNA_mlmc_term, strangNA_var_term] = mlmc_strang_na(T,N,M,u0,v0,coefs,payoff)[0::2]
        strangNA_mlmc_sum = strangNA_mlmc_sum + strangNA_mlmc_term
        strangNA_ml_var[l] = strangNA_var_term
        
        # Levels
        levels[l] = l
        timesteps[l] = T/N
        strangF_mlmc_hist[l] = strangF_mlmc_sum
        strangR_mlmc_hist[l] = strangR_mlmc_sum
        strangG_mlmc_hist[l] = strangG_mlmc_sum
        strangNA_mlmc_hist[l] = strangNA_mlmc_sum
    # Compute the error at each level
    strangF_error = np.abs(strangF_mlmc_hist - exact_val)
    strangR_error = np.abs(strangR_mlmc_hist - exact_val)
    strangG_error = np.abs(strangG_mlmc_hist - exact_val)
    strangNA_error = np.abs(strangNA_mlmc_hist - exact_val)
    # Record errors and variances for this iteration
    out_array_variances[0, iteration, :] = strangF_ml_var
    out_array_variances[1, iteration, :] = strangR_ml_var
    out_array_variances[2, iteration, :] = strangG_ml_var
    out_array_variances[3, iteration, :] = strangNA_ml_var
    out_array_errors[0, iteration, :] = strangF_error
    out_array_errors[1, iteration, :] = strangR_error
    out_array_errors[2, iteration, :] = strangG_error
    out_array_errors[3, iteration, :] = strangNA_error

# PLOTTING

# Variance reduction by level
strangF_v_mean = np.log(out_array_variances[0, :, :]).mean(axis = 0)[1:]
strangR_v_mean = np.log(out_array_variances[1, :, :]).mean(axis = 0)[1:]
strangG_v_mean = np.log(out_array_variances[2, :, :]).mean(axis = 0)[1:]
strangNA_v_mean = np.log(out_array_variances[3, :, :]).mean(axis = 0)[1:]
plt.plot(levels[1::],np.log(120) -1.35*levels[1::],linestyle='-.', color = 'black')
plt.plot(levels[1::],strangF_v_mean,marker='^', linewidth=2.2, color = colors[3],linestyle = '--')
plt.plot(levels[1::],strangR_v_mean,marker='v',linewidth=2.2, color = colors[4],linestyle=':')
plt.plot(levels[1::],strangG_v_mean,marker='s',linewidth=2.2, color = colors[2])
plt.plot(levels[1::],strangNA_v_mean,marker='o',linewidth=2.2, color = colors[5],linestyle='-')
plt.title('Multilevel Variance')
plt.xlabel('Level')
plt.xticks(range(1,L))
plt.ylabel('log(var)')
plt.legend(['Gradient -1.35','Strang T', 'Strang F','Strang G', 'Strang NA'])
plt.savefig('feller_fakes_variance_.png')
plt.close()


# plotting error vs level
strangF_e_mean = np.log(out_array_errors[0, :, :]).mean(axis = 0)
strangR_e_mean = np.log(out_array_errors[1, :, :]).mean(axis = 0)
strangG_e_mean = np.log(out_array_errors[2, :, :]).mean(axis = 0)
strangNA_e_mean = np.log(out_array_errors[3, :, :]).mean(axis = 0)
strangF_e_std = np.log(out_array_errors[0, :, :]).std(axis = 0)
strangR_e_std = np.log(out_array_errors[1, :, :]).std(axis = 0)
strangG_e_std = np.log(out_array_errors[2, :, :]).std(axis = 0)
strangNA_e_std = np.log(out_array_errors[3, :, :]).std(axis = 0)
plt.plot(levels[0::],(-1*levels[0::]+0.4),linestyle='-.', color = 'black')
plt.plot(levels[0::],strangR_e_mean,marker='^',linewidth=2.2, color = colors[3],linestyle = '--')
plt.plot(levels[0::],strangF_e_mean,marker='v',linewidth=2.2, color = colors[4],linestyle=':')
plt.plot(levels[0::],strangG_e_mean,marker='s',linewidth=2.2, color = colors[2])
plt.plot(levels[0::],strangNA_e_mean,marker='o',linewidth=2.2, color = colors[5],linestyle='-')
plt.plot(levels[0::],-1.38*levels[0::],linestyle='-.',color='black')
plt.fill_between(levels[0::], strangR_e_mean-strangR_e_std, strangR_e_mean+strangR_e_std, alpha = 0.3, facecolor=colors[3])
plt.fill_between(levels[0::], strangF_e_mean-strangF_e_std, strangF_e_mean+strangF_e_std, alpha = 0.3, facecolor=colors[4])
plt.fill_between(levels[0::], strangG_e_mean-strangG_e_std, strangG_e_mean+strangG_e_std, alpha = 0.3, facecolor=colors[2])
plt.fill_between(levels[0::], strangNA_e_mean-strangNA_e_std, strangNA_e_mean+strangNA_e_std, alpha = 0.3, facecolor=colors[5])
plt.title('Empirical Error')
plt.xlabel('Level')
plt.xticks(range(0,L))
plt.ylabel('log(error)')
plt.legend(['Gradients -1 and -1.38','Strang T', 'Strang F','Strang G', 'Strang NA'])
plt.savefig('feller_fakes_error_.png')
plt.close()