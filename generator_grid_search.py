import os

from src.aux_functions import make_pretty

from tabulate import tabulate
import torch
from src.LevyGAN import LevyGAN
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

test_config = {
    "bm_dim": 4,
    "do_timeing": False,
}

levy_gan = LevyGAN(test_config)

training_config = {
    "bm_dim": 4,
    "trainer_type": "pcf",
    "num_iters": 2500,  # for Chen training and CF training
    "optimizer": "Adam",
    "lrG": 0.000008,
    "lrD": 0.0001,
    "num_discr_iters": 3,
    "beta1": 0.2,
    "beta2": 0.97,
    "training_bsz": 2 * 4096,
    "testing_frequency": 100,
    "compute_joint_error": False,
    "print_reports": True,  # whether to print out reports as we go
    "descriptor": "",  # will appear in the filename of both the graph and the parameter dictionary
    "chen_penalty_alpha": 1.0,  # The coefficient for chen training (only relevant for Rotational_inv_LevyGAN)
    "rotation_penalty_alpha": 1.0,  # The coefficient for rotation training(only relevant for Rotational_inv_LevyGAN)
    "antisym_mult": 0.0,  # The multiplier for antisymmetric penalty (only relevant for PairNet Generator)
    "custom_lrs": {  # for Chen training and CF training
        0: (0.001, 0.01),
        # 200: (0.000008, 0.0001),
        # 400: (0.0000008, 0.00001),
        1000: (0.0001, 0.001),
        1500: (0.00001, 0.0001),
        2000: (0.000008, 0.0001),
    },
}


tester_config = {
    "bm_dim": 4,
    "test_bsz": 2**20,
    "joint_wass_dist_bsz": 5000,
    "num_tests_for_lowdim": 6,
    "BM_fixed_increment_whole": [1.0, -0.5, -1.2, -0.3, 0.7, 0.2, -0.9, 0.1, 1.7],
    "should_draw_graphs": True,
    "do_timeing": False,
}


small_table = [[["-LKY -NSZ L1", "-LKY +NSZ L1"]], [["+LKY -NSZ L1", "+LKY +NSZ L1"]]]

small_table = [
    [
        ["LKY=0.01 NSZ=4 alpha=0.1", "LKY=0.01 NSZ=4 alpha=1"],
        ["LKY=0.01 NSZ=8 alpha=0.1", "LKY=0.01 NSZ=8 alpha=1"],
    ],
    [
        ["LKY=0.2 NSZ=4 alpha=0.1", "LKY=0.2 NSZ=4 alpha=1"],
        ["LKY=0.2 NSZ=8 alpha=0.1", "LKY=0.2 NSZ=8 alpha=1"],
    ],
]

legend = tabulate(small_table)
# embedding = Embedding_layer(10)
levy_gan.init_trainer(training_config)
levy_gan.init_tester(tester_config)
# trainer.discriminator = get_discriminator(discr_config)


def get_result(dis_batch, lie_degree, leaky_slope, noise_size, alpha):
    gen_config = {
        "use_pair_net": True,
        "bm_dim": 4,
        "noise_size": noise_size,  # size of latent space
        "num_layers": 3,
        "hidden_dim": 16,
        "activation": ("leaky", leaky_slope),  # or 'relu'
        "batch_norm": True,
        "pairnet_bn": True,
        "do_bridge_flipping": True,  # "bridge_flipping", otherwise off
        "use_mixed_noise": False,  # Uses noise from several distributions. Gives bad results for now...
        "use_attention": False,
        "enforce_antisym": False,
        "reinject_h": False,
        "gen_dict_saving_on": True,
    }

    discr_config = {
        "bm_dim": 4,
        "CF_Discr_hidden_dim": 10,
        "discr_type": "path_characteristic",
        # "grid_characteristic", 'gaussian_characteristic', 'iid_gaussian_characteristic', 'embedded_characteristic', 'path_characteristic'
        "discr_measure": "Gaussian",  # "Gaussian", 'Cauchy'
        "coeff_batch": dis_batch,
        "lie_degree": lie_degree,
        "discr_dict_saving_on": False,
        "early_stopping": True,
        "loss_norm": 1,
    }

    levy_gan.trainer.chen_penalty_alpha = alpha

    levy_gan.init_trainer(training_config)

    res = []
    init_seed = 3407
    for seed in range(4):
        random_seed = init_seed + seed

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        levy_gan.init_discriminator(discr_config)
        levy_gan.init_generator(gen_config)
        levy_gan.tester.reset_test_results()

        # trainer.embedding_model = embedding.to(trainer.device)
        descriptor = f"{alpha}ALPHA_{noise_size}NSZ_{dis_batch}_DIS_BATCH_{lie_degree}_LIE_seed_{random_seed}_{levy_gan.generator.net_description}"
        training_config["descriptor"] = descriptor
        print(f"start training generator for {descriptor}, seed={random_seed}")

        levy_gan.fit(save_models=True)
        print(f"best score report: {levy_gan.tester.test_results['best score report']}")

        if not res:
            res = levy_gan.tester.test_results["best score"]
            print(res)
        else:
            print(res)
            res = min(res, levy_gan.tester.test_results["best score"])
    return make_pretty(res, decimal_places=4)


def get_small_table(dis_batch, lie_degree):
    table = [
        [
            [
                get_result(dis_batch, lie_degree, leaky_slope, noise_size, alpha)
                for alpha in [1.0]
            ]
            for noise_size in [4, 8]
        ]
        for leaky_slope in [0.01, 0.2]
    ]
    return tabulate(table)


def compute_big_row(dis_batch):
    row = [dis_batch]
    results = [get_small_table(dis_batch, lie_degree) for lie_degree in [5]]
    row.extend(results)
    return row


def get_full_table():
    headers = ["pcf batch size \ lie degree", 5]
    table = [compute_big_row(dis_batch) for dis_batch in [128]]
    res = tabulate(table, headers)
    print(res)
    return res


full_results = get_full_table()

contents_to_save = (
    f"Legend for small teble entries:\n\n"
    f"{legend} \n\nRESULTS:\n\n"
    f"{full_results}"
)

# find an unoccupied filename
i = 0
while os.path.exists(f"./model_saves/dim=4_discr_iters_3_pcf_grid_results_{i}.txt"):
    i += 1

file = open(f"./model_saves/dim=4_discr_iters_3_pcf_grid_results_{i}.txt", "w")
n = file.write(contents_to_save)
file.close()
