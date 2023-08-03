import importlib
import os.path
import timeit
from statistics import mean
from pathlib import Path
import scipy
import torch.cuda
from hyperopt import STATUS_OK

import configs_folder.configs as configs
from src.evaluation.evaluation import *
from src.model.Generator import get_generator, Generator, PairNetGenerator
from src.model.discriminator import Discriminator
from src.train.LevyGAN_tester import LevyGAN_Tester
from src.train.LevyGAN_trainer import BaseTrainer, get_trainer


class LevyGAN:
    def __init__(self, trainer_conf: dict = None):
        if trainer_conf is None:
            importlib.reload(configs)
            conf = configs.config
        else:
            conf = trainer_conf

        # ============ Model config ===============
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.conf = conf
        conf['device'] = self.device
        self.bm_dim = conf['bm_dim']
        self.levy_dim = int((self.bm_dim * (self.bm_dim - 1)) // 2)

        self.generator: Generator = None

        self.discriminator: Discriminator.Characteristic_Discriminator = None

        self.tester: LevyGAN_Tester = None

        self.trainer: BaseTrainer = None


    def init_generator(self, gen_config: dict = None):
        conf = gen_config if (gen_config is not None) else self.conf
        assert conf['bm_dim'] == self.bm_dim or conf['use_pair_net'], f"bm_dim should be {self.bm_dim}"
        conf['bm_dim'] = self.bm_dim
        self.generator = get_generator(conf)

    def init_discriminator(self, discr_config: dict = None):
        conf = discr_config if (discr_config is not None) else self.conf
        conf['bm_dim'] = self.bm_dim
        self.discriminator = Discriminator.get_discriminator(conf)

    def init_tester(self, test_config: dict = None):
        conf = test_config if (test_config is not None) else self.conf
        assert conf['bm_dim'], f"bm_dim must be provided for the tester"
        self.tester = LevyGAN_Tester(conf)

    def init_trainer(self, train_config: dict = None):
        conf = train_config if (train_config is not None) else self.conf
        conf['bm_dim'] = self.bm_dim
        self.trainer = get_trainer(conf)

    def init_all(self, config: dict = None):
        if self.generator is None:
            self.init_generator(config)
        if self.discriminator is None:
            self.init_discriminator(config)
        if self.tester is None:
            self.init_tester(config)
        if self.trainer is None:
            self.init_trainer(config)


    def fit(self, generator=None, discriminator=None, trainer=None, tester=None, save_models=True):
        # Initialize all the modules
        if generator is None:
            generator = self.generator
        if trainer is None:
            trainer = self.trainer
        if discriminator is None:
            discriminator = self.discriminator
        if tester is None:
            tester = self.tester
        # For saving functionalities
        tester.which_trainer = trainer.__class__.__name__

        assert (generator.bm_dim == self.bm_dim or
                isinstance(generator, PairNetGenerator)), f"generator.bm_dim should be {self.bm_dim}"
        assert discriminator.bm_dim == self.bm_dim, f"discriminator.bm_dim should be {self.bm_dim}"

        # Load the training config from the trainer
        tr_conf = trainer.conf

        # ============== Training config ===============
        # Number of training epochs using classical training
        num_iters = tr_conf['num_iters']

        if 'testing_frequency' in tr_conf:
            testing_frequency = tr_conf['testing_frequency']
        else:
            testing_frequency = 100

        # 'Adam' of 'RMSProp'
        which_optimizer = tr_conf['optimizer']

        # Learning rate for optimizers
        lr_g = tr_conf['lrG']
        lr_d = tr_conf['lrD']

        # Beta hyperparam for Adam optimizers
        beta1 = tr_conf['beta1']
        beta2 = tr_conf['beta2']

        wgt_dec = tr_conf['gen_weight_decay'] if 'gen_weight_decay' in tr_conf else 0

        if which_optimizer == 'Adam':
            opt_g = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(beta1, beta2),
                                     weight_decay=wgt_dec)
            opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(beta1, beta2))
        else:
            print("You sure you want RMSProp?!")
            opt_g = torch.optim.RMSprop(generator.parameters(), lr=lr_g)
            opt_d = torch.optim.RMSprop(discriminator.parameters(), lr=lr_d)

        training_bsz = tr_conf['training_bsz']

        compute_joint_w2_loss = tr_conf['compute_joint_error']

        tester.descriptor = tr_conf['descriptor']
        if 'print_reports' in tr_conf:
            tester.print_reports = tr_conf['print_reports']
        if 'should_draw_graphs' in tr_conf:
            tester.should_draw_graphs = tr_conf['should_draw_graphs']

        # Early stopping setup
        tester.test_results['min sum'] = float('inf')

        # For graphing
        tester.reset_training_score_tracker()

        # Attach the models to device
        generator.to(self.device)
        discriminator.to(self.device)

        iters = 0
        for i in range(num_iters):

            if 'custom_lrs' in tr_conf:
                lrs = tr_conf['custom_lrs']
                if iters in lrs:
                    lr_g_cust, lr_d_cust = lrs[iters]
                    opt_g = torch.optim.Adam(generator.parameters(), lr=lr_g_cust,
                                             betas=(beta1, beta2), weight_decay=wgt_dec)
                    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d_cust, betas=(beta1, beta2))
                    print(f"its: {iters} changed lrs to G: {lr_g_cust}, D: {lr_d_cust}")

            discriminator.zero_grad()
            generator.zero_grad()

            loss_g = trainer.step_fit(iters, generator, opt_g, discriminator, opt_d, tester.test_results)

            if iters % testing_frequency == 0:

                tester.do_tests(generator,
                                discriminator,
                                iters,
                                comp_joint_err=compute_joint_w2_loss,
                                hard_loading=False,
                                save_models=save_models)

                # Check whether there is a model collapse, if so, stop training
                if isinstance(discriminator, Discriminator.IID_Gaussian_Characteristic_Discriminator):
                    if discriminator.early_stopping and discriminator.levy_logvar[0].item() < -0.5:
                        print('Discriminator collapsed at iteration: ', iters)
                        break
            iters += 1

        self.tester.wrap_up_work(generator=generator,
                                 discriminator=discriminator,
                                 save_models=save_models)
