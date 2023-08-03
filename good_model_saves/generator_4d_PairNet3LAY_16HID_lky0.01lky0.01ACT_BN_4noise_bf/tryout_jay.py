import configs_folder.configs as configs
import torch
import torch.nn as nn
import numpy as np
from src.train.LevyGAN_trainer import BaseTrainer
from src.model.discriminator.Discriminator import get_discriminator
from src.model.Generator import Generator, PairNetGenerator
from src.LevyGAN import LevyGAN
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main(config):
    """
    This is the main function that triggers both training and evaluation module rather than using the jupyter notebook.
    We can also plug wandb here to log the results in the website.
    :param configs
    :return:
    """

    test_config = {
        'bm_dim': 4,
        'do_timeing': False,
    }

    levy_gan = LevyGAN(test_config)

    training_config = {
        'bm_dim': 4,
        'trainer_type': 'pcf',
        'num_iters': 2500,  # for Chen training and CF training
        'optimizer': 'Adam',
        'lrG': 0.000008,
        'lrD': 0.0001,
        'num_discr_iters': 3,
        'beta1': 0.2,
        'beta2': 0.97,
        'training_bsz': 2*4096,
        'testing_frequency': 100,
        'compute_joint_error': False,
        'print_reports': True,  # whether to print out reports as we go
        'descriptor': '',  # will appear in the filename of both the graph and the parameter dictionary
        'chen_penalty_alpha': 0.01,  # The coefficient for chen training (only relevant for Rotational_inv_LevyGAN)
        'rotation_penalty_alpha': 1.,  # The coefficient for rotation training(only relevant for Rotational_inv_LevyGAN)
        'antisym_mult': 0.,  # The multiplier for antisymmetric penalty (only relevant for PairNet Generator)
        # 'custom_lrs': {  # for Chen training and CF training
        #     1000: (0.000008, 0.0001),
        #     2000: (0.000001, 0.0001),
        #     4000: (0.00001, 0.00005),
        # },
        # 'custom_lrs': {  # for Chen training and CF training
        #     0: (0.001, 0.01),
        #     # 200: (0.000008, 0.0001),
        #     # 400: (0.0000008, 0.00001),
        #     1000: (0.0001, 0.001),
        #     1500: (0.00001, 0.0001),
        #     # 4000: (0.00001, 0.00005),
        # },
        'custom_lrs': {  # for Chen training and CF training
            0: (0.001, 0.01),
            # 200: (0.000008, 0.0001),
            # 400: (0.0000008, 0.00001),
            1000: (0.0001, 0.001),
            1500: (0.00001, 0.0001),
            2000: (0.000008, 0.0001),
        }
    }

    levy_gan.init_trainer(training_config)

    gen_config = {
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

    levy_gan.init_generator(gen_config)

    discr_config = {
        'bm_dim': 4,
        'CF_Discr_hidden_dim': 10,
        'discr_type': "path_characteristic",
        # "grid_characteristic", 'gaussian_characteristic', 'iid_gaussian_characteristic', 'embedded_characteristic', 'path_characteristic'
        'discr_measure': "Gaussian",  # "Gaussian", 'Cauchy'
        'coeff_batch': 128,
        'lie_degree': 3,
        'discr_dict_saving_on': False,
        'early_stopping': True,
        'loss_norm': 1
    }

    levy_gan.init_discriminator(discr_config)

    # trainer = Reciprocal_LevyGAN(embedding_model=embedding, trainer_conf=test_config)
    # trainer = Rotational_inv_LevyGAN(trainer_conf=test_config)
    # trainer.init_tester()

    tester_config = {
        'bm_dim': 4,
        'test_bsz': 2**20,
        'joint_wass_dist_bsz': 5000,
        'num_tests_for_lowdim': 6,
        'BM_fixed_increment_whole': [1.0, -0.5, -1.2, -0.3, 0.7, 0.2, -0.9, 0.1, 1.7],
        'should_draw_graphs': True,
        'do_timeing': False,
    }

    levy_gan.init_tester(tester_config)

    descriptor = f"{training_config['chen_penalty_alpha']}ALPHA_{gen_config['noise_size']}NSZ_{discr_config['coeff_batch']}_DIS_BATCH_{discr_config['lie_degree']}_LIE_seed_{3407}_{levy_gan.generator.net_description}"
    print(descriptor)
    description = 'embedded_chen_penalty_alpha_1_rotation_penalty_alpha_1_seed_3409_8NSZ_PairNet3LAY_64HID_lky0.2lky0.2ACT_BN_best_4mom'
    description = 'embedded_chen_penalty_alpha_1.0_rotation_penalty_alpha_1.0_seed_3408_4NSZ_PairNet3LAY_16HID_lky0.2lky0.2ACT_BN_best_4mom'
    discr_description = 'embedded_chen_penalty_alpha_0.1_rotation_penalty_alpha_0.1_seed_3431_8NSZ_PairNet3LAY_64HID_lky0.2lky0.2ACT_BN_best__scr'

    init_seed = 3407
    for seed in range(1):
        random_seed = init_seed + seed

        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        levy_gan.init_discriminator(discr_config)
        levy_gan.init_generator(gen_config)
        levy_gan.tester.reset_test_results()
        # levy_gan.discriminator = get_discriminator(discr_config)
        # levy_gan.discriminator.load_dict(3, discr_description)
        # generator = PairNetGenerator(gen_conf=gen_config)
        # levy_gan.generator.load_dict(1,description)
        # from src.model.network import Embedding_layer
        # embedding = Embedding_layer(10)
        # levy_gan.trainer.embedding_model = embedding.to(levy_gan.device)
        # levy_gan.generator = generator
        descriptor = f"embedded_chen_penalty_alpha_{training_config['chen_penalty_alpha']}" \
                     f"_rotation_penalty_alpha_{training_config['rotation_penalty_alpha']}" \
                     f"_seed_{random_seed}_{gen_config['noise_size']}" \
                     f"NSZ_{levy_gan.generator.net_description}"
        training_config['descriptor'] = descriptor
        # print(f"start training generator for {descriptor}, seed={random_seed}")

        levy_gan.fit(save_models=True)
        print(f"best score report: {levy_gan.tester.test_results['best score report']}", random_seed)
        # print(levy_gan.discriminator.levy_logvar)
        # if not res:
        #     res = trainer.test_results['best score']
        #     print(res)
        # else:
        #     print(res)
        #     res = min(res, trainer.test_results['best score'])


def test():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    gen_config = {
        'bm_dim': 3,
        'noise_size': 8,  # size of latent space
        'num_layers': 3,
        'hidden_dim': 16,
        'activation': ('leaky', 0.01),  # or 'relu'
        'batch_norm': True,
        'do_bridge_flipping': True,  # "bridge_flipping", otherwise off
        'gen_dict_saving_on': True
    }

    generator = Generator(config=gen_config)
    generator.load_state_dict(torch.load(
        './model_saves/generator_3d_net3LAY_16HID_lky0.01ACT_BN_8noise_bf/gen_num1_seed_3409_8NSZ_3LAY_16HID_lky0.01ACT_BN_max_scr.pt'))
    generator.to(device)

    # x_real = torch.tensor(np.genfromtxt(f"samples/fixed_samples_{3}-dim{1}.csv", dtype=float, delimiter=',')).to(dtype=torch.float, device=device)
    x_real = torch.tensor(np.genfromtxt(f"samples/samples_{3}-dim.csv", dtype=float, delimiter=',')).to(
        dtype=torch.float, device=device)
    levy_real = x_real[:, gen_config['bm_dim']:]

    x_fake = generator.sample_fake_data(x_real[:, :gen_config['bm_dim']])

if __name__ == '__main__':
    # test()
    main(configs)