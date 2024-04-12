import copy
import math
from pathlib import Path

import torch.nn as nn
import torch
from torch.distributions import *
from math import sqrt


from src.aux_functions import fast_flipping, read_serial_number, Davie_gpu
from src.model.network import create_net


def get_generator(gen_conf: dict):
    return PairNetGenerator(gen_conf)


class Generator(nn.Module):
    def __init__(self, gen_conf: dict):
        super(Generator, self).__init__()
        # Define device
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # Dimension of noise to use for generator (including H)
        self.noise_size = gen_conf["noise_size"]
        # Set dimensions
        self.bm_dim = gen_conf["bm_dim"]
        self.levy_dim = int((self.bm_dim * (self.bm_dim - 1)) // 2)
        # Saving parameters
        self.dict_saves_folder = None
        self._serial_num = -1
        self.dict_saving_on = (
            gen_conf["gen_dict_saving_on"]
            if ("gen_dict_saving_on" in gen_conf)
            else False
        )
        # Network initialisation
        self.layer_list = nn.ModuleList()
        self.netG = nn.Linear(1, 1)  # just a dummy so load_dict and save_dict work
        self.net_description = "empty"
        # Performs bridge flipping if true
        self.do_bridge_flipping = (
            gen_conf["do_bridge_flipping"] if "do_bridge_flipping" in gen_conf else True
        )

        self.bernoulli_sampler = Bernoulli(torch.tensor([0.5]))
        (
            self.levy_indices,
            self.h_indices,
            self.triu_indices,
        ) = self.generate_levy_h_triu_indices(self.bm_dim)

    # The following four functions provide functionality for saving a trained generator, and also loading a trained generator from a file
    def init_dict_saves_folder(self):
        if (self.dict_saves_folder is not None) and (self._serial_num > 0):
            return
        bf_str = "_bf" if self.do_bridge_flipping else ""
        self.dict_saves_folder = f"generator_{self.bm_dim}d_{self.net_description}_{self.noise_size}noise{bf_str}"
        Path(f"model_saves/{self.dict_saves_folder}/").mkdir(
            parents=True, exist_ok=True
        )
        if self._serial_num < 1:
            self._serial_num = read_serial_number(self.dict_saves_folder)

    def change_serial_number(self, new_serial: int):
        assert new_serial >= 1
        self.init_dict_saves_folder()
        next_available_serial = read_serial_number(self.dict_saves_folder)
        if new_serial > next_available_serial:
            print(
                f"Warning! This serial number is too high, setting it to {next_available_serial}"
            )
            self._serial_num = next_available_serial
        else:
            self._serial_num = new_serial

    def load_dict(
        self, serial_num_to_load: int = -1, descriptor: str = "", filename: str = ""
    ):
        self.init_dict_saves_folder()

        if filename == "":
            if serial_num_to_load < 1:
                sn = self._serial_num
            else:
                sn = serial_num_to_load
            folder_name = f"model_saves/{self.dict_saves_folder}/"
            filename = folder_name + f"gen_num{sn}_{descriptor}.pt"

        params = torch.load(filename, map_location=self.device)
        for i, layer in enumerate(self.layer_list):
            layer.load_state_dict(params[i])
        self.netG_from_layer_list()

    def save_dict(self, report: str = "", descriptor: str = ""):
        if not self.dict_saving_on:
            return

        self.init_dict_saves_folder()

        params = [copy.deepcopy(layer.state_dict()) for layer in self.layer_list]
        filename = f"model_saves/{self.dict_saves_folder}/summary_file.txt"
        line_header = f"{self._serial_num} {descriptor}"
        summary = f"{line_header}: {report} \n"

        with open(filename, "r+") as summary_file:
            lines = summary_file.readlines()
            summary_file.seek(0)

            flag = False
            for i in range(len(lines)):
                line_header_from_file = lines[i].split(":")[0]
                if line_header == line_header_from_file:
                    lines[i] = summary
                    flag = True
                    break

            if not flag:
                lines.append(summary)

            summary_file.writelines(lines)
            summary_file.truncate()

        folder_name = f"model_saves/{self.dict_saves_folder}/"
        torch.save(params, folder_name + f"gen_num{self._serial_num}_{descriptor}.pt")

    def netG_from_layer_list(self):
        raise NotImplementedError

    def get_rademacher(self, shape):
        return 2 * (self.bernoulli_sampler.sample(shape) - 0.5).to(self.device)

    def forward(self, input):
        raise NotImplementedError

    def sample_fake_data(self, input, bb_from_h=False):
        """Sample from generator in eval mode"""
        self.eval()
        with torch.no_grad():
            if bb_from_h:
                out = self.generate_bb(h_in=input).detach()
            else:
                out = self.forward(input).detach()
        self.train()
        return out

    def generate_bb(self, bsz: int = 1, h_in: torch.Tensor = None, concat_hb=True):
        raise NotImplementedError


class PairNetGenerator(Generator):
    def __init__(self, gen_conf):
        super(PairNetGenerator, self).__init__(gen_conf)
        # Initialise network
        self.layer_list, self.net_description = create_net(
            gen_conf, 2 * self.noise_size, 1
        )
        # Set descriptor for saving
        self.net_description = "PairNet" + self.net_description
        # Type of nosie to use for generator. Defaults to Gaussian
        self.noise_types = (
            gen_conf["noise_types"] if ("noise_types" in gen_conf) else []
        )
        if isinstance(self.noise_types, str):
            self.noise_types = [self.noise_types]
        self.start_time = 0

        # ======= Make distributions for noise ========
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        one = torch.ones((), dtype=torch.float, device=device)
        zero = torch.zeros((), dtype=torch.float, device=device)
        uni_dist = Uniform(low=(-1 * one), high=one)
        logi_scale = (1 / (2 * math.pi)) * one
        transforms = [SigmoidTransform().inv, AffineTransform(loc=0, scale=logi_scale)]
        logistic_dist = TransformedDistribution(uni_dist, transforms)
        self.distros = {
            "ber": Bernoulli(probs=0.5 * one),
            "lap": Laplace(loc=zero, scale=one),
            "cauchy": Cauchy(loc=zero, scale=1),
            "uni": uni_dist,
            "logi": logistic_dist,
        }

    def get_rademacher(self, shape):
        return 2 * (self.distros["ber"].sample(shape) - 0.5).to(self.device)

    def netG_from_layer_list(self):
        return

    def generate_levy_h_triu_indices(self, new_bm_dim: int):
        """Generates the indices for all objects required for generator
        Args:
            new_bm_dim (int): may be different from training bm_dim

        Returns:
            _type_: _description_
        """
        noise_indices = torch.arange(self.noise_size).view(1, 1, -1)
        triu_indices = torch.triu_indices(new_bm_dim, new_bm_dim, offset=1).permute(
            1, 0
        )
        levy_indices = torch.flatten(
            self.noise_size * triu_indices.unsqueeze(2) + noise_indices,
            start_dim=1,
            end_dim=2,
        ).to(self.device)
        h_indices = self.noise_size * torch.arange(new_bm_dim).to(self.device)
        return levy_indices, h_indices, triu_indices

    def get_latent_vector(self, bsz, bm_dim=None):
        """
        Generates the latent vector aka noise (info about BM). This includes h = sqrt(1 / 12) * noise[:, h_indices].
        If self.use_mixed_noise == True, the output will be composed of several distributions.
        Otherwise, just a Gaussian.
        Args:
            bsz:

        Returns:
            torch.Tensor: Latent vector of shape (bsz, noise_size*bm_dim)
        """

        if bm_dim is None:
            bm_dim = self.bm_dim

        if len(self.noise_types) == 0:
            out = torch.randn(
                (bsz, self.noise_size * bm_dim), dtype=torch.float, device=self.device
            )
            return out

        def get_single_dim_noise(bsz):
            dims_per_type = 2
            num_gaussian_dims = self.noise_size - dims_per_type * len(self.noise_types)
            if num_gaussian_dims < 1:
                raise ValueError(
                    "Too many noise types. At least one dim must be left for a gaussian"
                )
            gauss = torch.randn(
                (bsz, num_gaussian_dims), dtype=torch.float, device=self.device
            )

            noises = [gauss]
            for noise_type in self.noise_types:
                sample = self.distros[noise_type].sample((bsz, dims_per_type))
                noises.append(sample)
                # print(f"MAKING SOME NOISEEEEE {noise_type}")

            return torch.cat(noises, dim=1)

        whole_noise_list = [get_single_dim_noise(bsz) for i in range(bm_dim)]
        out = torch.cat(whole_noise_list, dim=1)
        assert out.shape == (bsz, self.noise_size * bm_dim)
        return out

    def forward(self, input):
        """

        Args:
            input (torch.Tensor): the Brownian increment

        Returns:
            torch.Tensor: (BM || Levy area)
        """

        bm = input
        bsz = input.shape[0]
        new_bm_dim = input.shape[1]
        # Generates space-time levy area and space-space levy area of brownian bridge
        h, b = self.generate_bb(bsz=bsz, new_bm_dim=new_bm_dim, concat_hb=False)

        # Perform bridge-flipping
        if not self.do_bridge_flipping:
            # Last rademacher will ensure odd moments of levy are zero
            levy = Davie_gpu(bm, device=self.device, h_in=h, bb=b)
            last_rademacher = self.get_rademacher((b.shape[0],))
        else:
            r = self.get_rademacher((bsz, new_bm_dim + 1)).squeeze()
            r_for_flipping, last_rademacher = torch.split(r, [new_bm_dim, 1], dim=1)
            levy = fast_flipping(bm, b, h, r_in=r_for_flipping, device=self.device)
            del h, b, r

        levy = torch.mul(last_rademacher, levy)
        out = torch.cat((bm, levy), dim=1)
        return out

    def generate_bb(
        self, bsz: int = 1, h_in: torch.Tensor = None, new_bm_dim=None, concat_hb=True
    ):
        """
        Generates bb (space-space bridge levy area) conditional on H (space-time bridge levy area)
        Args:
            bsz: batch size (is ignored if h_in is supplied)
            h_in: space-time bridge levy area H
            concat_hb: will concat H and bb if True (else returns a tuple)
        Returns:
            torch.Tensor | (torch.Tensor, torch.Tensor): (H || bb) or (H, bb)
        """
        # Set all indices correctly for generation
        if h_in is not None:
            h = h_in
            bsz = h_in.shape[0]
            new_bm_dim = h_in.shape[1]
            levy_indices, h_indices, triu_indices = self.generate_levy_h_triu_indices(
                new_bm_dim
            )
            noise = self.get_latent_vector(bsz)
            noise[:, h_indices] = h
        else:
            if new_bm_dim is None:
                new_bm_dim = self.bm_dim
            if new_bm_dim == self.bm_dim:
                levy_indices, h_indices, triu_indices = (
                    self.levy_indices,
                    self.h_indices,
                    self.triu_indices,
                )
            else:
                (
                    self.levy_indices,
                    self.h_indices,
                    self.triu_indices,
                ) = self.generate_levy_h_triu_indices(new_bm_dim)
                levy_indices, h_indices, triu_indices = (
                    self.levy_indices,
                    self.h_indices,
                    self.triu_indices,
                )
            noise = self.get_latent_vector(bsz, new_bm_dim)
            if self.do_bridge_flipping:
                noise[:, h_indices] *= sqrt(1 / 12)
            h = noise[:, h_indices]
        # Set dimensions
        new_levy_dim = int((new_bm_dim * (new_bm_dim - 1)) // 2)
        self.bm_dim = new_bm_dim
        self.levy_dim = new_levy_dim
        # flatten array
        x = torch.flatten(noise[:, levy_indices], start_dim=0, end_dim=1).detach()
        # Forward pass through network
        for i, layer in enumerate(self.layer_list):
            x = layer(x, new_levy_dim)
        b = x.view(-1, new_levy_dim)
        if concat_hb:
            return torch.cat((h, b), dim=1)
        else:
            return h, b

    def generate_MC_samples(self, M: int, N: int, dt: float):
        """generates samples with shape M x N x (bm_dim + levy_dim) over the time interval dt

        Args:
            M (int): number of MC paths
            N (int): number of timesteps
            dt (float): timestep
        """
        self.eval()
        with torch.no_grad():
            bm = torch.randn(
                (M * N, self.bm_dim), dtype=torch.float, device=self.device
            )
            out = self.forward(bm).detach()
            out[:, : self.bm_dim] *= sqrt(dt)
            out[:, self.bm_dim :] *= dt
            out = out.view(M, N, -1)
        # self.train()
        return out
