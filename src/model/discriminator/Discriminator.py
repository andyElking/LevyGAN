import copy
from pathlib import Path

import torch
import torch.nn as nn
from src.aux_functions import read_serial_number
from src.model.discriminator.characteristic_function import get_fake_characteristic, get_real_characteristic
from src.model.discriminator.unitary_representation import development_layer



def get_discriminator(discr_conf: dict):
    """Returns the desired type of discriminator; this can either be a version of the regular characteristic discriminator or the unitary characteristic discriminator.

    Args:
        discr_conf (dict): config describing the discriminator setup.

    Returns:
        : discriminator
    """
    discriminator = DISCRIMINATOR[discr_conf['discr_type']](discr_conf)
    
    return discriminator


class Characteristic_Discriminator(nn.Module):
    def __init__(self, discr_conf: dict):
        super(Characteristic_Discriminator, self).__init__()
        # Get the true joint characteristic function
        self.real_characteristic = get_real_characteristic()
        # Set desired dimensions
        self.bm_dim = discr_conf['bm_dim']
        self.levy_dim = int((self.bm_dim * (self.bm_dim - 1)) // 2)
        self.total_dim = self.bm_dim + self.levy_dim
        # Set device type
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # Number of coefficients at which to calculate characteristic function
        self.coeff_batch = 64
        if 'coeff_batch' in discr_conf:
            self.coeff_batch = discr_conf['coeff_batch']
        # Measure with which to sample coefficients
        self.measure = 'Gaussian'
        if 'discr_measure' in discr_conf:
            self.measure = discr_conf['discr_measure']
        # CF distance is an L^p distance
        self.p = 1
        if 'loss_norm' in discr_conf:
            self.p = discr_conf['loss_norm']
        self.fake_characteristic = None
        # Saving setup
        self.dict_saves_folder = None
        self._serial_num = -1
        self.dict_saving_on = discr_conf['discr_dict_saving_on'] if ('discr_dict_saving_on' in discr_conf) else False
    
    # The following four functions provide functionality to both save the discriminator and also load a pre-trained discriminator from a file
    def init_dict_saves_folder(self):
        if (self.dict_saves_folder is not None) and (self._serial_num > 0):
            return
        self.dict_saves_folder = f'discriminator_{self.measure}measure_{self.net_architecture}_{self.bm_dim}d'
        Path(f"model_saves/{self.dict_saves_folder}/").mkdir(parents=True, exist_ok=True)
        if self._serial_num < 1:
            self._serial_num = read_serial_number(self.dict_saves_folder)

    def change_serial_number(self, new_serial: int):
        assert new_serial >= 1
        self.init_dict_saves_folder()
        next_available_serial = read_serial_number(self.dict_saves_folder)
        if new_serial > next_available_serial:
            print(f"Warning! This serial number is too high, setting it to {next_available_serial}")
            self._serial_num = next_available_serial
        else:
            self._serial_num = new_serial

    def load_dict(self, serial_num_to_load: int = -1, descriptor: str = ""):
        self.init_dict_saves_folder()

        if serial_num_to_load < 1:
            sn = self._serial_num
        else:
            sn = serial_num_to_load
        folder_name = f'model_saves/{self.dict_saves_folder}/'
        filename = folder_name + f'discr_num{sn}_{descriptor}.pt'
        self.load_state_dict(torch.load(filename, map_location=self.device))

    def save_dict(self, report: str = "", descriptor: str = ""):
        if not self.dict_saving_on:
            return

        self.init_dict_saves_folder()

        params = copy.deepcopy(self.state_dict())
        filename = f'model_saves/{self.dict_saves_folder}/summary_file.txt'
        line_header = f"{self._serial_num} {descriptor}"
        summary = f"{line_header}: {report} \n"

        with open(filename, 'r+') as summary_file:
            lines = summary_file.readlines()
            summary_file.seek(0)

            flag = False
            for i in range(len(lines)):
                line_header_from_file = lines[i].split(':')[0]
                if line_header == line_header_from_file:
                    lines[i] = summary
                    flag = True
                    break

            if not flag:
                lines.append(summary)

            summary_file.writelines(lines)
            summary_file.truncate()

        folder_name = f'model_saves/{self.dict_saves_folder}/'
        torch.save(params, folder_name + f'discr_num{self._serial_num}_{descriptor}.pt')
    
    def loss_function(self, fake_characteristic, real_characteristic, p = 2):
        """
        Computes the L^p distance between real and fake characteristic function
        """
        
        difference = (fake_characteristic - real_characteristic)
        lp_norm = torch.linalg.norm(difference, ord = p)/torch.pow(torch.tensor(self.coeff_batch, device = self.device), 1/p)
        return lp_norm

    def init_fake_characteristic(self, input):
        """Returns the characteristic function defined by the empirical measure of the input
        Args:
            input (torch.tensor): input
        """
        self.fake_characteristic = get_fake_characteristic(input, just_real = True)

    def sample_coefficients(self):
        raise NotImplementedError
    
    def get_loss(self, fake_characteristic, real_characteristic):
        """Computes the estimated CF-distance between the true characteristic function and characteristic defined by the generator

        Args:
            fake_characteristic : characteristic function defined by a finite sample from the generator
            real_characteristic : true characteristic function

        Returns:
            torch.tensor: estimated distance
        """
        # Samples coefficients at which to evaluate the characteristic function
        coefficients = self.sample_coefficients()
        # Evaluate characteristic functions
        char_fake = fake_characteristic(coefficients)
        char_real = real_characteristic(coefficients, self.bm_dim)
        # Get loss
        D_loss = self.loss_function(char_fake, char_real, self.p)
        return D_loss

    def true_char_diff(self, X: torch.tensor):
        """Computes the CF distance to the true characteristic function

        Args:
            X (torch.tensor): sample from generator

        Returns:
            torch.tensor: loss
        """
        if X is not None:
            self.init_fake_characteristic(X)
        res = self.get_loss(self.fake_characteristic, self.real_characteristic)
        return res

    def empirical_char_diff(self, input_1, input_2, just_real_1 = False, just_real_2 = False):
        """Computes the CF distance between the empirical measures of two inputs

        Args:
            input_1 (torch.tensor): input 1
            input_2 (torch.tensor): input_2
            just_real_1 (bool, optional): if True, discards the imaginary part of CF function. Defaults to False.
            just_real_2 (bool, optional): if True, discards the imaginary part of CF function. Defaults to False.

        Returns:
            torch.tensor: loss
        """
        empirical_characteristic_1 = get_fake_characteristic(input_1, just_real = just_real_1)
        empirical_characteristic_2 = get_fake_characteristic(input_2, just_real = just_real_2)
        coefficients = self.sample_coefficients()
        char_1 = empirical_characteristic_1(coefficients)
        char_2 = empirical_characteristic_2(coefficients)
        D_loss = self.loss_function(char_1, char_2, self.p)
        return D_loss


class Grid_Characteristic_Discriminator(Characteristic_Discriminator):
    '''
    We parametrize an empirical distribution on the coefficient space
    '''
    def __init__(self, discr_conf: dict):
        super(Grid_Characteristic_Discriminator, self).__init__(discr_conf)

        self.coefficients = nn.Parameter(torch.empty(self.coeff_batch, self.levy_dim), requires_grad=True)
        nn.init.kaiming_normal_(self.coefficients)

    def sample_coefficients(self):
        bm_coefficients = torch.empty([self.coeff_batch, self.bm_dim], device=self.device)
        bm_coefficients.normal_()
        coefficients = torch.cat([bm_coefficients, self.coefficients], -1)
        return coefficients


class IID_Gaussian_Characteristic_Discriminator(Characteristic_Discriminator):
    '''
    We assume the coefficients follows a IID Gaussian distribution coordinatewise where the variances are learnt
    '''
    def __init__(self, discr_conf: dict):
        super(IID_Gaussian_Characteristic_Discriminator, self).__init__(discr_conf)

        # self.levy_mean = torch.empty(1, 1), requires_grad=True
        self.levy_logvar = nn.Parameter(torch.empty(1, 1), requires_grad=True)
        nn.init.kaiming_normal_(self.levy_logvar)
        self.early_stopping = discr_conf['early_stopping'] if 'early_stopping' in discr_conf else False
        # nn.init.constant_(self.levy_logvar, 0.0)
        
    def sample_coefficients(self):
        levy_coefficients = torch.empty([self.coeff_batch, self.levy_dim], device=self.device)
        # Simulate the coefficients for BM, this distribution is not learnt by the model
        bm_coefficients = torch.empty([self.coeff_batch, self.bm_dim], device=self.device)
        bm_coefficients.normal_()
        if self.measure == 'Cauchy':
            levy_coefficients.cauchy_()
        else:
            levy_coefficients.normal_()

        levy_coefficients = torch.pow(torch.exp(self.levy_logvar), 0.5) * levy_coefficients
        if self.measure == 'Cauchy':
            levy_coefficients = torch.clamp(levy_coefficients, -100, 100)
        coefficients = torch.cat([bm_coefficients, levy_coefficients], -1)
        return coefficients


class Gaussian_Characteristic_Discriminator(Characteristic_Discriminator):
    '''
    We assume the coefficients follows a Gaussian distribution where the mean and variance are optimized
    '''
    def __init__(self, discr_conf: dict):
        super(Gaussian_Characteristic_Discriminator, self).__init__(discr_conf)

        self.levy_mean = nn.Parameter(torch.empty(1, self.levy_dim))
        self.levy_logvar = nn.Parameter(torch.empty(1, self.levy_dim))
        nn.init.kaiming_normal_(self.levy_mean)
        nn.init.kaiming_normal_(self.levy_logvar)
        self.early_stopping = discr_conf['early_stopping'] if 'early_stopping' in discr_conf else False

    def sample_coefficients(self):
        levy_coefficients = torch.empty([self.coeff_batch, self.levy_dim], device=self.device)
        # Simulate the coefficients for BM, this distribution is not learnt by the model
        bm_coefficients = torch.empty([self.coeff_batch, self.bm_dim], device=self.device)
        bm_coefficients.normal_()
        if self.measure == 'Cauchy':
            levy_coefficients.cauchy_()
        else:
            levy_coefficients.normal_()

        levy_coefficients = self.levy_mean + torch.pow(torch.exp(self.levy_logvar), 0.5) * levy_coefficients
        if self.measure == 'Cauchy':
            levy_coefficients = torch.clamp(levy_coefficients, -100, 100)
        coefficients = torch.cat([bm_coefficients, levy_coefficients], -1)
        return coefficients


class UCF_Discriminator(nn.Module):
    def __init__(self, discr_conf):
        super(UCF_Discriminator, self).__init__()
        # Number of linear maps to optimise
        self.coeff_batch = discr_conf['coeff_batch'] if 'coeff_batch' in discr_conf else 64
        # Degree of lie algebra to use
        self.lie_degree = discr_conf['lie_degree'] if 'lie_degree' in discr_conf else 1
        # Set dimensions
        self.bm_dim = discr_conf['bm_dim']
        self.levy_dim = int((self.bm_dim * (self.bm_dim - 1)) // 2)
        self.total_dim = self.bm_dim + self.levy_dim
        # Linear maps to optimise
        self.unitary_representation = development_layer(input_size=self.total_dim, hidden_size=self.lie_degree,
                                                        channels=self.coeff_batch)
        # Saving paramters
        self.dict_saves_folder = None
        self._serial_num = -1
        self.dict_saving_on = discr_conf['discr_dict_saving_on'] if ('discr_dict_saving_on' in discr_conf) else False
        for param in self.unitary_representation.parameters():
            param.requires_grad = True

    def init_dict_saves_folder(self):
        if (self.dict_saves_folder is not None) and (self._serial_num > 0):
            return
        self.dict_saves_folder = f'ucf_discriminator_{self.lie_degree}_LIE_{self.bm_dim}d'
        Path(f"model_saves/{self.dict_saves_folder}/").mkdir(parents=True, exist_ok=True)
        if self._serial_num < 1:
            self._serial_num = read_serial_number(self.dict_saves_folder)


    @staticmethod
    def HS_norm(X: torch.tensor, Y: torch.Tensor):
        """_summary_

        Args:
            X (torch.Tensor): (C,m,m) complexed valued
        """
        if len(X.shape) == 4:

            m = X.shape[-1]
            X = X.reshape(-1, m, m)

        else:
            pass
        D = torch.bmm(X, torch.conj(Y).permute(0, 2, 1))
        return torch.sqrt((torch.einsum('bii->b', D)).mean().real)


    def empirical_char_diff(self, X1: torch.tensor, X2: torch.tensor) -> torch.float:
        """distance measure given by the Hilbert-schmidt inner product
           d_hs(A,B) = trace[(A-B)(A-B)*]**(0.5)
           measure = \integral d_hs(\phi_{x1}(m),\phi_{x2}(m)) dF_M(m)
           let m be the linear map sampled from F_M(m)
        Args:
            X1 (torch.tensor): time series samples with shape (N_1,d)
            X2 (torch.tensor): time series samples with shape (N_2,d)
        Returns:
            torch.float: distance measure between two batch of samples
        """
        # print(X1.shape)
        unit1, unit2 = self.unitary_representation(X1), self.unitary_representation(X2)

        # initial_dev = self.unitary_development_initial()
        CF1, CF2 = unit1.mean(0), unit2.mean(0)

        return self.HS_norm(CF1-CF2, CF1-CF2)

    def save_dict(self, report: str = "", descriptor: str = ""):
        if not self.dict_saving_on:
            return

        self.init_dict_saves_folder()

        params = copy.deepcopy(self.unitary_representation.state_dict())
        filename = f'model_saves/{self.dict_saves_folder}/summary_file.txt'
        line_header = f"{self._serial_num} {descriptor}"
        summary = f"{line_header}: {report} \n"

        with open(filename, 'r+') as summary_file:
            lines = summary_file.readlines()
            summary_file.seek(0)

            flag = False
            for i in range(len(lines)):
                line_header_from_file = lines[i].split(':')[0]
                if line_header == line_header_from_file:
                    lines[i] = summary
                    flag = True
                    break

            if not flag:
                lines.append(summary)

            summary_file.writelines(lines)
            summary_file.truncate()

        folder_name = f'model_saves/{self.dict_saves_folder}/'
        torch.save(params, folder_name + f'discr_num{self._serial_num}_{descriptor}.pt')


DISCRIMINATOR = {'grid_characteristic': Grid_Characteristic_Discriminator,
                 'gaussian_characteristic': Gaussian_Characteristic_Discriminator,
                 'iid_gaussian_characteristic': IID_Gaussian_Characteristic_Discriminator,
                 'u_characteristic': UCF_Discriminator
                 }