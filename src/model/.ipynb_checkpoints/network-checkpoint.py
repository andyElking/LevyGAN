import torch.nn as nn
import torch


def generator_main(conf: dict):
    gen_net = conf['gen_net']
    bm_dim = conf['bm_dim']
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    levy_dim = int((bm_dim * (bm_dim - 1)) // 2)
    noise_size = conf['noise_size']
    if gen_net == 1:
        layers = nn.Sequential(
            nn.Linear(bm_dim + noise_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, levy_dim)
        ).to(device=device)
    if gen_net == 2:
        layers = nn.Sequential(
            nn.Linear(bm_dim + noise_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            # nn.Linear(1024,1024),
            # nn.BatchNorm1d(1024),
            # nn.ReLU(),

            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, levy_dim)
        ).to(device=device)
    if gen_net == 3:
        layers = nn.Sequential(
            nn.Linear(bm_dim+noise_size,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128,levy_dim)
        ).to(device=device)
    if gen_net == 4:
        layers = nn.Sequential(
            nn.Linear(bm_dim + noise_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, levy_dim)
        ).to(device=device)
    if gen_net == 5:
        layers = nn.Sequential(
            nn.Linear(bm_dim + noise_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, levy_dim)
        ).to(device=device)
    if gen_net == 6:
        layers = nn.Sequential(
            nn.Linear(bm_dim+noise_size,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024,levy_dim)
        ).to(device=device)
    if gen_net == 7:
        layers = nn.Sequential(
            nn.Linear(bm_dim + noise_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, levy_dim)
        ).to(device=device)
    if gen_net == 9:
        leakyReLU_slope = 0.1
        if 'gen_leaky_slope' in conf:
            leakyReLU_slope = conf['gen_leaky_slope']
        layers = nn.Sequential(
            nn.Linear(bm_dim + noise_size, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(leakyReLU_slope),

            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(leakyReLU_slope),

            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(leakyReLU_slope),

            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(leakyReLU_slope),

            nn.Linear(128, levy_dim)
        ).to(device=device)
    if gen_net == 10:
        leakyReLU_slope = 0.2
        if 'gen_leaky_slope' in conf:
            leakyReLU_slope = conf['gen_leaky_slope']
        layers = nn.Sequential(
            nn.Linear(bm_dim + noise_size, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(leakyReLU_slope),

            nn.Linear(256, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(leakyReLU_slope),

            nn.Linear(256, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(leakyReLU_slope),

            nn.Linear(256, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(leakyReLU_slope),

            nn.Linear(256, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(leakyReLU_slope),

            nn.Linear(256, levy_dim)
        ).to(device=device)

    return layers



