import torch
import torch.cuda
import numpy as np
from src.model.Generator import PairNetGenerator
from src.LevyGAN import LevyGAN
from src.evaluation.evaluation import full_evaluation
from src.evaluation.shuffle_prod import nth_moments
import argparse

print(torch.cuda.is_available())


def model_training(
    gan_config=None,
    gen_config=None,
    training_config=None,
    discr_config=None,
    tester_config=None,
):
    """
    This is the main function that does training.
    :param configuration dictionaries for the GAN, generator, discriminator, trainer and tester.
    Examples can be found in configs.py
    :return: trained model will be saved in /model_saves the training plots will be saved in /graphs.
    """

    # If not provided, we set some default configurations
    if not gan_config:
        gan_config = {
            "bm_dim": 4,
        }

    # Training configuration
    if not training_config:
        training_config = {
            "bm_dim": 4,
            "trainer_type": "ucf",
            "num_iters": 1500,  # for Chen training and CF training
            "optimizer": "Adam",
            "lrG": 0.000008,
            "lrD": 0.0001,
            "num_discr_iters": 3,
            "beta1": 0.2,
            "beta2": 0.97,
            "training_bsz": 2048,
            "testing_frequency": 100,
            "compute_joint_error": False,
            "print_reports": True,  # whether to print out reports as we go
            "descriptor": "",  # will appear in the filename of both the graph and the parameter dictionary
            "chen_penalty_alpha": 0.01,  # The coefficient for chen training (only relevant for
            "custom_lrs": {  # for Chen training and CF training
                0: (0.001, 0.01),
                1000: (0.0001, 0.001),
                1500: (0.00001, 0.0001),
                2000: (0.000008, 0.0001),
            },
        }

    # Generator configuration
    if not gen_config:
        gen_config = {
            "use_pair_net": True,
            "bm_dim": 4,
            "noise_size": 4,  # size of latent space
            "num_layers": 3,
            "hidden_dim": 16,
            "activation": ("leaky", 0.01),  # or 'relu'
            "batch_norm": True,
            "pairnet_bn": True,
            "do_bridge_flipping": True,  # "bridge_flipping", otherwise off
            "gen_dict_saving_on": True,
        }

    # Discriminator configuration
    if not discr_config:
        discr_config = {
            "bm_dim": 4,
            "discr_type": "u_characteristic",
            "coeff_batch": 128,
            "lie_degree": 3,
            "discr_dict_saving_on": False,
            "early_stopping": True,
            "loss_norm": 1,
        }

    # Tester configuration
    if not tester_config:
        tester_config = {
            "bm_dim": 4,
            "test_bsz": 2**20,
            "joint_wass_dist_bsz": 5000,
            "num_tests_for_lowdim": 6,
            "BM_fixed_increment_whole": [
                1.0,
                -0.5,
                -1.2,
                -0.3,
                0.7,
                0.2,
                -0.9,
                0.1,
                1.7,
            ],
            "should_draw_graphs": True,
            "do_timeing": False,
        }

    # Initialize the GAN and its corresponding components
    levy_gan = LevyGAN(gan_config)
    levy_gan.init_trainer(training_config)
    levy_gan.init_generator(gen_config)
    levy_gan.init_discriminator(discr_config)
    levy_gan.init_tester(tester_config)

    # Reset the test results and set the seed
    random_seed = 3407
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    levy_gan.tester.reset_test_results()

    # Name this run
    descriptor = (
        f"seed_{random_seed}_{gen_config['noise_size']}"
        f"{levy_gan.generator.net_description}"
    )
    training_config["descriptor"] = descriptor
    # Start training
    levy_gan.fit(save_models=False)
    print(
        f"best score report: {levy_gan.tester.test_results['best score report']}",
        random_seed,
    )


def model_evaluation(generator_dir, gen_config):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    generator = PairNetGenerator(gen_conf=gen_config)
    params = torch.load(generator_dir)

    # Load the model
    for i, layer in enumerate(generator.layer_list):
        layer.load_state_dict(params[i])
    generator.to(device)
    generator.eval()

    bm_dim = gen_config["bm_dim"]

    four_mom = nth_moments(bm_dim=bm_dim, n=4).to(device)

    # Load "real_data" which needs to be generated using the julia package. Here is an example:
    x_real = torch.tensor(
        np.genfromtxt("samples/samples_4-dim.csv", dtype=float, delimiter=",")
    ).to(dtype=torch.float, device=device)
    # Run the evaluation
    loss_dict = full_evaluation(
        x_real, generator, gen_config, device, real_fourth_moments=four_mom
    )


if __name__ == "__main__":
    # test()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", type=str, default="train", help="choose from TimeGAN,RCGAN,TimeVAE"
    )
    args = parser.parse_args()
    if args.task == "train":
        # model_training(configs.gan_config, configs.gen_config, configs.training_config, configs.discr_config,
        # configs.tester_config)
        model_training()
