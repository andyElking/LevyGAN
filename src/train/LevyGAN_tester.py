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

class LevyGAN_Tester:
    def __init__(self, config_in: dict = None):
        if config_in is None:
            importlib.reload(configs)
            conf = configs.config
        else:
            conf = config_in

        # ============ Model config ===============
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.conf = conf
        conf['device'] = self.device
        self.bm_dim = conf['bm_dim']
        self.levy_dim = int((self.bm_dim * (self.bm_dim - 1)) // 2)

        # ============ Testing config ============
        self.num_tests = conf['num_tests_for_lowdim'] if 'num_tests_for_lowdim' in conf else 1
        if 3 <= self.bm_dim <= 5 and self.num_tests > 6:
            self.num_tests = 6
        elif self.bm_dim > 5:
            self.num_tests = 1

        if 'which_samples_to_load' in conf:
            self.which_samples_to_load = conf['which_samples_to_load']
            self.num_tests = len(self.which_samples_to_load)
        else:
            self.which_samples_to_load = [i+1 for i in range(self.num_tests)]

        self.max_test_bsz = conf['test_bsz']
        self.test_bsz = min(2**16, self.max_test_bsz)
        self.joint_wass_dist_bsz = conf['joint_wass_dist_bsz']

        # These are never loaded until self.load_samples is first called (e.g. at the top of do_tests)
        self.single_coord_labels = None
        self.fixed_true_levy = None
        self.fixed_bm_increments = None
        self.true_fourth_moments = None
        self.joint_labels = None

        self.test_results = {
            'marginal w2 loss': [],
            'joint wass errors': [float('inf')],
            'best joint errors': [float('inf')],
            'st dev error': -1.0,
            'loss d': [float('inf')],
            'gradient norm': 0.0,
            'best score': float('inf'),
            'best score report': '',
            'fourth_mom_errors': [],
            'best_max_4mom_errors': float('inf')
        }

        self.training_score_tracker = {
            'marginal_w2_losses_tt': [],
            'discriminative_losses_tt': [],
            'fourth_mom_losses_tt': [],
            'iters_xticks': []
        }

        self.print_reports = conf['print_reports'] if 'print_reports' in conf else True
        self.should_draw_graphs = conf['should_draw_graphs'] if 'should_draw_graphs' in conf else True
        self.descriptor = None
        self.which_trainer = None

    def reset_training_score_tracker(self):
        self.training_score_tracker = {
            'marginal_w2_losses_tt': [],
            'discriminative_losses_tt': [],
            'fourth_mom_losses_tt': [],
            'iters_xticks': []
        }

    def load_samples(self, hard=False):
        """
        Load the samples for testing, unless they have already been loaded.
        :param hard: Reload them regardless of whether they have been loaded already.
        :return:
        """
        if not hard and (self.fixed_true_levy is not None):
            return

        self.single_coord_labels = []
        self.fixed_true_levy = []
        self.fixed_bm_increments = []
        self.joint_labels = []
        self.true_fourth_moments = []
        for i in self.which_samples_to_load:
            filename = f"samples/fixed_samples_{self.bm_dim}-dim{i}.csv"
            if self.num_tests == 1 and not os.path.exists(filename):  # the not lowdim case
                filename = f"samples/fixed_samples_{self.bm_dim}-dim.csv"

            # first read just the bm (first row)
            bm_data = np.genfromtxt(filename, dtype=float, delimiter=',', usecols=range(self.bm_dim), max_rows=1)
            fixed_bm_increment = torch.tensor(bm_data, dtype=torch.float, device=self.device)
            bm = make_pretty(fixed_bm_increment.cpu().tolist(), decimal_places=2)

            # now load just the levy area
            levy_cols = range(self.bm_dim, self.bm_dim + self.levy_dim)
            levy_data = np.genfromtxt(filename, dtype=float, delimiter=',',
                                      max_rows=self.max_test_bsz, usecols=levy_cols)

            true_levy = torch.tensor(levy_data, dtype=torch.float, device=self.device)
            self.fixed_true_levy.append(true_levy)
            self.fixed_bm_increments.append(fixed_bm_increment)

            self.joint_labels.append(bm)
            self.single_coord_labels += list_pairs(self.bm_dim, bm)
            self.true_fourth_moments.append(fourth_moments(levy_data))

    def reload_testing_config(self, conf: dict):
        self.num_tests = conf['num_tests_for_lowdim']

        self.max_test_bsz = conf['test_bsz']
        self.joint_wass_dist_bsz = conf['joint_wass_dist_bsz']

        self.BM_fixed_increment_whole = conf['BM_fixed_increment_whole']
        self.BM_fixed_increment = torch.tensor(self.BM_fixed_increment_whole, device=self.device)[
                                  :self.bm_dim].unsqueeze(1).transpose(1, 0)
        self.BM_fixed_increment = self.BM_fixed_increment.expand((self.max_test_bsz, self.bm_dim))

        self.load_samples(hard=True)

        self.reset_test_results()

        self.print_reports = conf['print_reports'] if 'print_reports' in conf else True
        self.should_draw_graphs = conf['should_draw_graphs'] if 'should_draw_graphs' in conf else True

    def reset_test_results(self):
        self.test_results = {
            'marginal w2 loss': [],
            'joint wass errors': [float('inf')],
            'best joint errors': [float('inf')],
            'st dev error': -1.0,
            'loss d': [float('inf')],
            'gradient norm': 0.0,
            'best score': float('inf'),
            'best score report': '',
            'fourth_mom_errors': [],
            'best_max_4mom_errors': float('inf')
        }
        self.test_bsz = 2**16

    def do_tests(self, generator, discriminator=None, iters: int = None, comp_joint_err=False, hard_loading=False, save_models=False):
        self.load_samples(hard_loading)
        # The main evaluation function, track test metric during the traning.
        # TODO: Add other test metric, such as marginal distribution, correlation metric etc.

        generator.eval()
        if discriminator is not None:
            discriminator.eval()

        # Here we calculate the losses which will be store in arrays: marginal_w2_losses, joint_wass_losses, discriminative_losses
        # Record the marginal wise
        joint_wass_losses = []
        marginal_w2_losses = []
        # discriminative_losses = []
        fourth_mom_losses = []

        for i in range(self.num_tests):
            # Test Wasserstein error for fixed BM
            levy_fixed_true = self.fixed_true_levy[i]
            true_4moms = self.true_fourth_moments[i]
            levy_fixed_true = levy_fixed_true[:self.test_bsz]
            fixed_bm = self.fixed_bm_increments[i]
            marginal_w2_loss, _, st_dev_err, mom4_loss = \
                compute_fixed_losses(generator=generator,
                                     config=self.conf,
                                     input_true_data=levy_fixed_true,
                                     true_fourth_moments=true_4moms,
                                     input_true_bm=fixed_bm)

            marginal_w2_losses += marginal_w2_loss
            fourth_mom_losses.append(mom4_loss)
            if comp_joint_err:
                joint_w2_loss = compute_joint_losses(generator=generator,
                                                     config=self.conf,
                                                     input_true_data=levy_fixed_true[0],
                                                     device=self.device,
                                                     bm_dim=self.bm_dim)
                joint_wass_losses.append(joint_w2_loss.item())

        # Track everything just in case
        self.test_results['joint wass errors'] = joint_wass_losses
        self.test_results['marginal w2 loss'] = marginal_w2_losses
        self.test_results['fourth_mom_errors'] = fourth_mom_losses

        score = self.model_score(c=0.0)

        # self.test_bsz only increases when the model gets good enough for a larger bsz to matter
        scores_and_sizes = [(1.8, 14), (0.9, 16), (0.4, 18)]

        if self.test_bsz < self.max_test_bsz:
            for scr_threshold, exponent in scores_and_sizes:
                if self.test_bsz == 2**exponent and score < scr_threshold:
                    self.test_bsz = min(self.max_test_bsz, 2**(exponent+2))
                    # print(f"INCREASED test_bsz to {self.test_bsz}")
                    self.do_tests(generator, discriminator, iters, comp_joint_err=comp_joint_err, save_models=save_models)
                    break

        flag_for_saving_dict = ''
        save_score = ''

        if score < self.test_results['best score']:
            self.test_results['best score'] = make_pretty(score)
            report = make_report(self, add_line_break=False)
            self.test_results['best score report'] = report
            flag_for_saving_dict = f'best__scr'
            save_score = str(make_pretty(score))

        max_4mom_errors = mean(fourth_mom_losses)
        if max_4mom_errors < self.test_results['best_max_4mom_errors']:
            self.test_results['best_max_4mom_errors'] = max_4mom_errors
            flag_for_saving_dict = f'best_4mom'
            save_score = str(make_pretty(max_4mom_errors))

        report = make_report(self, short=True, add_line_break=False)

        if save_models and flag_for_saving_dict != '':
            self.save_current_dicts(generator,
                                    discriminator,
                                    report=report,
                                    descriptor=f"{self.descriptor}_{flag_for_saving_dict}")
            if self.print_reports:
                print(f"New {flag_for_saving_dict}: {save_score} at iter {iters}")

        if self.print_reports:
            report = make_report(self, iters=iters, short=True)
            print(report)
        else:
            print('.', end='')

        marginal_w2_loss = self.test_results['marginal w2 loss']
        self.training_score_tracker['marginal_w2_losses_tt'].append(marginal_w2_loss)
        self.training_score_tracker['fourth_mom_losses_tt'].append(self.test_results['fourth_mom_errors'])
        discriminative_loss = self.test_results['loss d']
        self.training_score_tracker['discriminative_losses_tt'].append(discriminative_loss)
        self.training_score_tracker['iters_xticks'].append(iters)

        generator.train()
        if discriminator is not None:
            discriminator.train()
        return

    def save_current_dicts(self, generator, discriminator, report: str, descriptor: str = ""):
        discriminator.save_dict(report=report, descriptor=descriptor)
        generator.save_dict(report=report, descriptor=descriptor)

    # Just a way to keep track of a combination of different error metrics. You can redefine it to something
    # else. It's just useful bc all graphs and dicts will have the score in their filename, so it's easy to see
    # which is best at first glance. Also needed for hyperparameter opt.
    def model_score(self, a: float = 100.0, b: float = 0.0, c: float = 0.0):
        res = 0.0
        res += a * mean(self.test_results['marginal w2 loss'])
        if c > 0.0 and len(self.test_results['joint wass errors']) > 0:
            res += c * self.levy_dim * mean(self.test_results['joint wass errors'])
        return res

    def draw_graphs(self, filename=None):
        if filename is None:
            filename = 'graphs/graph'
        draw_error_graphs(filename,
                          self.training_score_tracker['marginal_w2_losses_tt'],
                          self.single_coord_labels,
                          fourth_moments_through_training=self.training_score_tracker['fourth_mom_losses_tt'],
                          joint_labels=self.joint_labels,
                          discriminative_losses_through_training=self.training_score_tracker['discriminative_losses_tt'],
                          descriptor=f"{self.descriptor}_scr_{self.test_results['best score']}",
                          iters=self.training_score_tracker['iters_xticks'])

    def wrap_up_work(self, generator, discriminator, save_models=True):
        """
        This function is only called in the end of training,
        :return:
        """
        best_score = self.test_results['best score']
        best_4mom_errs = self.test_results['best_max_4mom_errors']
        if self.print_reports:
            print(f"Best score: {make_pretty(best_score)}, best 4mom errors: {make_pretty(best_4mom_errs)}")
        # Plot the losses against training iterations
        if self.should_draw_graphs:
            graph_filename = './graphs/dim={}_trainer={}_generator={}_discriminator={}'.format(
                self.bm_dim,
                self.which_trainer,
                generator.__class__.__name__,
                discriminator.__class__.__name__
            )
            # Create the path
            Path(graph_filename).mkdir(parents=True, exist_ok=True)
            graph_filename = graph_filename + '/graph'

            self.draw_graphs(graph_filename)
        if save_models:
            report = make_report(self, add_line_break=False)
            self.save_current_dicts(generator, discriminator, report=report, descriptor=f"{self.descriptor}_end_trn")