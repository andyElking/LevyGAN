import gc

import numpy as np
from math import sqrt
import ot
import torch
from src.train.Levy_CFGAN_trainer import LevyGAN
from src.aux_functions import *
import timeit

config = {
    'w dim': 3,
    'noise size': 61,
    'which generator': 10,
    'which discriminator': 10,
    'generator symmetry mode': 'Hsym',
    'leakyReLU slope': 0.2,
    'test bsz': 16384,
    'unfixed test bsz': 16384,
    'joint wass dist bsz': 8192,
    'num tests for 2d': 4,
    'W fixed whole': [1.0, -0.5, -1.2, -0.3, 0.7, 0.2, -0.9, 0.1, 1.7],
    'should draw graphs': True,
    'do timeing': False
}

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

w_dim = 3
a_dim = int((w_dim * (w_dim - 1)) // 2)
bsz = 262144
data = np.genfromtxt(f'samples/fixed_samples_3-dim_big.csv', dtype=float, delimiter=',', max_rows=bsz)
a_true = data[:, w_dim:(w_dim + a_dim)]
W = data[:, :w_dim]
W_torch = torch.tensor(W, dtype=torch.float, device=device)
print(data.shape)

def check_precision(_samples, _elapsed, name):
    a_samples = _samples[:, w_dim:(w_dim + a_dim)]
    err = [sqrt(ot.wasserstein_1d(a_true[:, i], a_samples[:, i], p=2)) for i in range(a_dim)]
    avg_err = sum(err) / len(err)
    joint_err = joint_wass_dist(a_true[:20000], a_samples[:20000])
    print(f"{name} time: {make_pretty(_elapsed,4)}, avg individual err: {make_pretty(avg_err, 6)}, joint err: {make_pretty(joint_err)}")

def check_precision_a(_a_samples, _elapsed, name):
    err = [sqrt(ot.wasserstein_1d(a_true[:, i], _a_samples[:, i], p=2)) for i in range(a_dim)]
    avg_err = sum(err)/len(err)
    joint_err = joint_wass_dist(a_true[:20000], _a_samples[:20000])
    print(f"{name} time: {make_pretty(_elapsed,4)}, avg individual err: {make_pretty(avg_err, 6)}, joint err: {make_pretty(joint_err)}")

# samples = np.genfromtxt(f'samples/mid_prec_fixed_samples_{w_dim}-dim.csv', dtype=float, delimiter=',', max_rows=bsz)
# elapsed = 0.111565
# check_precision(samples, elapsed, "julia p3")
#
# samples = np.genfromtxt(f'samples/p4_samples_3-dim.csv', dtype=float, delimiter=',', max_rows=bsz)
# elapsed = 0.162131
# check_precision(samples, elapsed, "julia p4")
#
# start_time = timeit.default_timer()
# samples = gen_2mom_approx(w_dim, bsz, _W = W)
# elapsed = timeit.default_timer() - start_time
# check_precision(samples, elapsed, "2mom_apx")
#

torch.cuda.empty_cache()
gc.collect()
T, M, S = generate_tms(w_dim, device)
h = sqrt(1 / 12) * torch.randn((bsz, w_dim), dtype=torch.float, device=device)
b = sqrt(1 / 12) * torch.randn((bsz, a_dim), dtype=torch.float, device=device)
wth = aux_compute_bmth(W_torch, h, S, T, w_dim)
a_wthmb = aux_compute_bmthmbb(wth, b, M, w_dim)
start_time = timeit.default_timer()
samples = mom4_gpu(w_dim, bsz, bm_in=W_torch)
elapsed = timeit.default_timer() - start_time
mom4_np = samples.cpu().numpy()
check_precision(mom4_np, elapsed, "4mom_apx")

# T, M, S = generate_tms(w_dim, device)
# h = sqrt(1 / 12) * torch.randn((bsz, w_dim), dtype=torch.float, device=device)
# b = sqrt(1 / 12) * torch.randn((bsz, a_dim), dtype=torch.float, device=device)
# wth = aux_compute_wth(W_torch, h, S, T, w_dim)
# a_wthmb = aux_compute_wthmb(wth, b, M, w_dim)
# x = a_wthmb[0,0].item()
#
# levG = LevyGAN(config_in=config, do_load_samples=False)
# levG.load_dicts(serial_num_to_load=22, descriptor="CLAS_max_scr")
# start_time = timeit.default_timer()
# samples = levG.eval(W_torch)
# elapsed = timeit.default_timer() - start_time
# a_gan_np = (samples.detach()[:, w_dim:(w_dim + a_dim)]).cpu().numpy()
# check_precision_a(a_gan_np, elapsed, "GAN     ")

# torch.cuda.empty_cache()
# gc.collect()
# T, M, S = generate_tms(w_dim, device)
# h = sqrt(1 / 12) * torch.randn((bsz, w_dim), dtype=torch.float, device=device)
# b = sqrt(1 / 12) * torch.randn((bsz, a_dim), dtype=torch.float, device=device)
# wth = aux_compute_wth(W_torch, h, S, T, w_dim)
# a_wthmb = aux_compute_wthmb(wth, b, M, w_dim)
#
# start_time = timeit.default_timer()
# h = sqrt(1 / 12) * torch.randn((bsz, w_dim), dtype=torch.float, device=device)
# b = sqrt(1 / 12) * torch.randn((bsz, a_dim), dtype=torch.float, device=device)
# wth = aux_compute_wth(W_torch, h, S, T, w_dim)
# a_wthmb = aux_compute_wthmb(wth, b, M, w_dim)
# elapsed = timeit.default_timer() - start_time
# a_wthmb_np = a_wthmb.cpu().numpy()
# check_precision_a(a_wthmb_np, elapsed, "F and L ")