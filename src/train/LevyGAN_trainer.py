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

"""The trainer class consists of a parent class, the BaseTrainer, and two derived classes. One is CF_Trainer, which trains the using the true joint characteristic function as the loss. The other is UCF_trainer, that trains using the unitary characeristic function combined with Chen training.
"""


def get_trainer(trainer_conf: dict):
    """Selects either CF_trainer or UCF_trainer
    """
    trainer = TRAINER[trainer_conf['trainer_type']](trainer_conf)
    return trainer


class BaseTrainer:

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

        # ============== Training config ===============
        # Number of training epochs using classical training
        self.training_bsz = conf['training_bsz'] if 'training_bsz' in conf else 2048
        # Number of generator updates per discriminator updates
        self.num_discr_iters = conf['num_discr_iters'] if 'num_discr_iters' in conf else 5

        self.start_time = timeit.default_timer()
        self.print_reports = conf['print_reports'] if 'print_reports' in conf else True

    def step_fit(self, iters, generator, opt_g, discriminator, opt_d, test_results):
        raise NotImplementedError


class CF_Trainer(BaseTrainer):
    """
    Derived class of the trainer, where training uses the analytical joint characteristic function as well as chen training
    """

    def __init__(self, trainer_conf: dict = None):
        super(CF_Trainer, self).__init__(trainer_conf)

        # Coefficient of chen loss
        if 'chen_penalty_alpha' in trainer_conf:
            self.chen_penalty_alpha = trainer_conf['chen_penalty_alpha']
        else:
            self.chen_penalty_alpha = 1.

    def generator_fit(self, generator, opt_g, discriminator):
        """Updates the generator
        """
        generator.zero_grad()
        # Generate increments of Brownian motion
        bm_increment = torch.randn((self.training_bsz, self.bm_dim), dtype=torch.float, device=self.device)
        # Generate fake Levy area
        fake_data = generator.forward(bm_increment)
        # Generate another sample of twice the batch size for Chen training
        with torch.no_grad():
            bm_increment = torch.randn((2 * self.training_bsz, self.bm_dim), dtype=torch.float, device=self.device)
            fake_data_detached = generator.forward(bm_increment).detach()
        # Perform one iteration of chen combine
        fake_data_chen = chen_combine(fake_data_detached, self.bm_dim)
        # Compute the loss using the joint characteristic function and the Chen loss
        loss_true = discriminator.true_char_diff(fake_data)
        loss_chen = discriminator.empirical_char_diff(fake_data, fake_data_chen, just_real_1=True)
        loss_g = loss_true + self.chen_penalty_alpha * loss_chen
        # Update step
        (loss_g).backward()
        opt_g.step()

        return fake_data, fake_data_chen, loss_g

    def discriminator_fit(self, discriminator, opt_d, fake_data, fake_data_chen):
        """Upates the discriminator

        Args:
            discriminator :
            opt_d :
            fake_data (torch.tensor): sample from the generator
            fake_data_chen (torch.tensor): independent sample of Chen-combined data from the generator
        """
        discriminator.zero_grad()
        # Detach all data from the generator
        fake_data_detached = fake_data.detach()
        fake_data_chen_detached = fake_data_chen.detach()
        # Compute CF loss and chen loss
        loss_d_true = discriminator.true_char_diff(fake_data_detached)
        loss_d_chen = discriminator.empirical_char_diff(fake_data_detached,
                                                        fake_data_chen_detached, just_real_1=True)
        loss_d = loss_d_true + self.chen_penalty_alpha * loss_d_chen
        # Update step
        (-loss_d).backward()
        opt_d.step()
        return

    def step_fit(self, iters, generator, opt_g, discriminator, opt_d, test_results):
        """
        Updates the generator and discriminator with the correct frequency
        """
        # Update generator
        fake_data, fake_data_chen, loss_g = self.generator_fit(
            generator, opt_g, discriminator)

        if self.num_discr_iters > 0:
            # Train generator N times per discriminator fit
            if iters % self.num_discr_iters == 0:
                self.discriminator_fit(discriminator, opt_d, fake_data, fake_data_chen)

        if self.num_discr_iters < 0:
            # Train discriminator N times per generator fit
            for discr_iter in range(-self.num_discr_iters):
                self.discriminator_fit(discriminator, opt_d, fake_data, fake_data_chen)
        test_results['loss d'] = [loss_g.detach().item()]

        return loss_g


class UCF_Trainer(BaseTrainer):
    """
    Derived class of the trainer, where training uses the unitary characteristic function for chen training
    """

    def __init__(self, trainer_conf: dict = None):
        super(UCF_Trainer, self).__init__(trainer_conf)

        # First we need to load the trained generator
        if 'chen_penalty_alpha' in trainer_conf:
            self.chen_penalty_alpha = trainer_conf['chen_penalty_alpha']
        else:
            self.chen_penalty_alpha = 1.

    def generator_fit(self, generator, opt_g, discriminator):
        """Updates the generator
        """
        generator.zero_grad()
        # Generate increments of Brownian motion
        bm_increment = torch.randn((self.training_bsz, self.bm_dim), dtype=torch.float, device=self.device)
        # Generate fake Levy area
        fake_data = generator.forward(bm_increment)
        # Generate another sample of twice the batch size for Chen training
        with torch.no_grad():
            bm_increment = torch.randn((2 * self.training_bsz, self.bm_dim), dtype=torch.float, device=self.device)
            fake_data_detached = generator.forward(bm_increment).detach()
        # Perform one iteration of chen combine
        fake_data_chen = chen_combine(fake_data_detached, self.bm_dim, chunking=True)
        # Compute Chen loss
        loss_g = discriminator.empirical_char_diff(fake_data, fake_data_chen)
        # Update step
        (self.chen_penalty_alpha * loss_g).backward()
        opt_g.step()

        return fake_data, fake_data_chen, loss_g

    def discriminator_fit(self, discriminator, opt_d, fake_data, fake_data_chen):
        """Upates the discriminator
        Args:
            discriminator :
            opt_d :
            fake_data (torch.tensor): sample from the generator
            fake_data_chen (torch.tensor): independent sample of Chen-combined data from the generator
        """
        discriminator.zero_grad()
        # Detach all data from the generator
        fake_data_detached = fake_data.detach()
        fake_data_chen_detached = fake_data_chen.detach()
        # Compute Chen loss
        loss_d = discriminator.empirical_char_diff(fake_data_detached,
                                                   fake_data_chen_detached)
        # Update Step
        (-self.chen_penalty_alpha * loss_d).backward()
        opt_d.step()
        return

    def step_fit(self, iters, generator, opt_g, discriminator, opt_d, test_results):
        """
        Updates the generator and discriminator with the correct frequency
        """
        # Update generator
        fake_data, fake_data_chen, loss_g = self.generator_fit(generator, opt_g, discriminator)

        if self.num_discr_iters > 0:
            # Train generator N times per discriminator fit
            if iters % self.num_discr_iters == 0:
                self.discriminator_fit(discriminator, opt_d, fake_data, fake_data_chen)

        if self.num_discr_iters < 0:
            # Train discriminator N times per generator fit
            for discr_iter in range(-self.num_discr_iters):
                self.discriminator_fit(discriminator, opt_d, fake_data, fake_data_chen)
        test_results['loss d'] = [loss_g.detach().item()]

        return loss_g


TRAINER = {'base': BaseTrainer,
           'ucf': UCF_Trainer,
           'cf': CF_Trainer
           }
