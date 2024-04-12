"""Configs file providing standard congigurations for the base class, trainer, tester, generator and discriminator as well as the options
"""

# Passed to base class
gan_config = {
    "bm_dim": 4,  # Dimension of brownian motion\
}

# Pass this to LevyGAN.init_generator
gen_config = {
    "use_pair_net": True,
    "bm_dim": 4,
    "noise_size": 4,  # size of latent space (including H)
    "noise_types": [],  # Different noise distributions: ['ber', 'lap', 'cauchy', 'uni', 'logi']. Defaults to Gaussian
    "num_layers": 3,
    "hidden_dim": 16,
    "activation": ("leaky", 0.1),  # 'relu', ('leaky', slope:float)
    "batch_norm": True,  # Use batch norm after each layer
    "pairnet_bn": True,  # Use pairnet batchnorm. batch_norm must be True for this option to matter
    "do_bridge_flipping": True,  # whether or not to do bridge flipping
    "use_mixed_noise": False,  # Uses noise from several distributions
    "gen_dict_saving_on": True,
}
# Pass this to LevyGAN.init_discriminator
discr_config = {
    "bm_dim": 4,
    "discr_type": "u_characteristic",  # "grid_characteristic", 'gaussian_characteristic', 'iid_gaussian_characteristic'
    "discr_measure": "Gaussian",  # "Gaussian", 'Cauchy', doesn't matter if u_characteristic selected
    "coeff_batch": 20,  # Number of points at which to evalute characteristic function
    "lie_degree": 5,  # Only matters for u_characteristic
    "discr_dict_saving_on": False,
    "loss_norm": 1,  # CF-distance is an L^p norm. Doesn't affect u_characteristic option
    "early_stopping": True,  # If True, stopping when the discriminator has collapsed (logvar < -0.5). Doesn't affect u_characteristic option
}

training_config = {
    "bm_dim": 4,
    "trainer_type": "ucf",  # ;'cf'
    "num_iters": 3000,  # For Chen training and CF training
    "optimizer": "Adam",
    "lrG": 0.000008,  # Learning rate of generator
    "lrD": 0.0001,  #  Learning rate of discrimator
    "num_discr_iters": 3,  # Number of generator training steps per discriminator training step
    "beta1": 0.2,  # Betas for optimisers
    "beta2": 0.97,  # Betas for optimisers
    "training_bsz": 4096,  # Training batch size
    "compute_joint_error": False,
    "testing_frequency": 100,  # Frequency with which to do testing
    "print_reports": True,  # Whether to print out reports as we go
    "descriptor": "",  # Will appear in the filename of both the graph and the parameter dictionary
    "chen_penalty_alpha": 0.1,  # The coefficient for chen training. Should be 1 if using ucf trainer
    "custom_lrs": {  # for Chen training and CF training
        0: (0.00001, 0.0001),
        2000: (0.00001, 0.00004),
        4000: (0.000004, 0.00002),
        8000: (0.000001, 0.000005),
    },
}


tester_config = {
    "bm_dim": 4,
    "test_bsz": 2**20,  # Number of test samples to use
    "joint_wass_dist_bsz": 5000,
    "num_tests_for_lowdim": 4,
    "which_samples_to_load": [
        1,
        2,
        8,
    ],  # manually select which samples to load (overrides num_tests_for_lowdim).
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
    ],  # Which increments to use for marginal tests
    "should_draw_graphs": True,  # Whether to draw graphs at end of training
}
