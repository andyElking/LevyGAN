import matplotlib.pyplot as plt
import torch
from src.model.Generator import PairNetGenerator
from src.aux_functions import MC_chen_combine, mom4_gpu, Davie_gpu_all, rademacher_GPU_dim2, rademacher_GPU_dim2_var
import time

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
#
# Setup levy area generator

def measure_time(bm_dim, number_paths):
    # Setup levy area generator
    gen_config = gen_config = {
                'use_pair_net': True,
                'bm_dim': 4,
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



    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    generator = PairNetGenerator(gen_conf=gen_config)
    params = torch.load("./good_model_saves/generator_4d_PairNet3LAY_16HID_lky0.01lky0.01ACT_BN_4noise_bf/gen_num5_best__scr.pt")

    # Load the model
    for i, layer in enumerate(generator.layer_list):
        layer.load_state_dict(params[i])
    generator.to(device)
    generator.eval()
    generator.do_bridge_flipping = True

    times_foster = []
    times_net = []
    times_davie = []

    steps = 100
    start_events_net = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
    end_events_net = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]


    generator.eval()
    # print(timeit.timeit(lambda: pattern(number), number= 1))
    for i in range(steps):
        tic = time.time()
        with torch.no_grad():
            start_events_net[i].record()
            bm = torch.randn(size=(number_paths, bm_dim), dtype=torch.float, device=device)
            x = generator(bm)
            end_events_net[i].record()
    torch.cuda.synchronize()

    times_net = [0.001*s.elapsed_time(e) for s, e in zip(start_events_net, end_events_net)]

    start_events_foster = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
    end_events_foster = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]

    for i in range(steps):
        tic = time.time()
        with torch.no_grad():
            start_events_foster[i].record()
            mom4_gpu(bm_dim, number_paths, device_to_use=device)
            end_events_foster[i].record()
    torch.cuda.synchronize()

    times_foster = [0.001*s.elapsed_time(e) for s, e in zip(start_events_foster, end_events_foster)]

    start_events_davie = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
    end_events_davie = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
    for i in range(steps):
        tic = time.time()
        with torch.no_grad():
            start_events_davie[i].record()
            Davie_gpu_all(bm_dim,number_paths)
            end_events_davie[i].record()
    torch.cuda.synchronize()

    times_davie = [0.001* s.elapsed_time(e) for s, e in zip(start_events_davie, end_events_davie)]

    # print(x[:10])
    times_foster = torch.tensor(times_foster)[10:]
    # print(times_foster)
    print("Foster: mean ", times_foster.mean(), "std ", times_foster.std())
    # print(times_net)
    times_net = torch.tensor(times_net)[10:]
    print("Net: mean ", times_net.mean(), "std ", times_net.std())

    times_davie = torch.tensor(times_davie)[10:]
    print("BN: mean ", times_davie.mean(), "std ", times_davie.std())

    return times_foster.mean().item(), times_foster.std().item(), times_net.mean().item(), times_net.std().item(), times_davie.mean().item(), times_davie.std().item()


if __name__ == '__main__':
    import pandas as pd
    bm_dims = [2,3,4,5,6,7,8]
    numbers_paths = [2**14, 2**16, 2**18]
    df = {}
    for bm_dim in bm_dims:
        for number_paths in numbers_paths:
            tfm, tfs, tnm, tns, tdm, tds = measure_time(bm_dim, number_paths)
            df['bm_dim_{}_np_{}'.format(bm_dim, number_paths)] = [tfm, tfs, tnm, tns, tdm, tds]
    df = pd.DataFrame(df)
    df.to_csv("./time_measurement.csv")