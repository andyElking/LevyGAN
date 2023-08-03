import copy
from pathlib import Path

import torch.nn as nn
import torch
from math import sqrt

from src.aux_functions import generate_tms, read_serial_number
from src.model.network import generator_main


class Generator(nn.Module):
    def __init__(self, config: dict):
        super(Generator, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.noise_size = config['noise_size']
        self.bm_dim = config['bm_dim']
        self.levy_dim = int((self.bm_dim * (self.bm_dim - 1)) // 2)
        self.num_flip_combinations = 2 ** self.bm_dim

        if 'do_bridge_flipping' in config:
            self.do_bridge_flipping = config['do_bridge_flipping']
        else:
            self.do_bridge_flipping = True

        self.net_architecture = config['gen_net'] if ('gen_net' in config) else 1
        self.netG = generator_main(config)
        self.T, self.M, self.S = generate_tms(_bm_dim=self.bm_dim, device=self.device)

        self.dict_saves_folder = None
        self._serial_num = -1
        self.dict_saving_on = config['gen_dict_saving_on'] if ('gen_dict_saving_on' in config) else False

    def init_dict_saves_folder(self):
        if (self.dict_saves_folder is not None) and (self._serial_num > 0):
            return
        bf_str = "_bf" if self.do_bridge_flipping else ""
        self.dict_saves_folder = f'generator_{self.bm_dim}d_net{self.net_architecture}_{self.noise_size}noise{bf_str}'
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
        filename = folder_name + f'gen_num{sn}_{descriptor}.pt'
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
        torch.save(params, folder_name + f'gen_num{self._serial_num}_{descriptor}.pt')


    def compute_bmth(self, bm_in: torch.Tensor, h_in: torch.Tensor):
        _bsz = bm_in.shape[0]
        assert bm_in.shape == (_bsz, self.bm_dim)
        assert h_in.shape == (_bsz, self.bm_dim)
        _H = torch.mul(self.S, h_in.view(_bsz, 1, self.bm_dim))
        _BMT = torch.tensordot(bm_in, self.T, dims=1)
        _BMTH = torch.flatten(torch.matmul(_H, _BMT), start_dim=0, end_dim=1)
        return _BMTH

    def compute_bmthmbb(self, bmth_in: torch.Tensor, b_in: torch.Tensor):
        _bsz = b_in.shape[0]
        assert bmth_in.shape == (_bsz * (2 ** self.bm_dim), self.levy_dim)
        assert b_in.shape == (_bsz, self.levy_dim)
        _B = b_in.view(1, _bsz, self.levy_dim)
        _MB = torch.flatten(torch.mul(self.M, _B).permute(1, 0, 2), start_dim=0, end_dim=1)
        return bmth_in + _MB

    def forward(self, input):
        bm = input
        bsz = input.shape[0]
        noise = torch.randn((bsz, self.noise_size), dtype=torch.float, device=self.device)
        if self.do_bridge_flipping:
            bsz = input.shape[0]
            h = sqrt(1 / 12) * torch.randn((bsz, self.bm_dim), dtype=torch.float, device=self.device)
            bmth = self.compute_bmth(bm, h).detach()
            x = torch.cat((noise, h), dim=1).detach()
            b = self.netG(x)
            levy = self.compute_bmthmbb(bmth, b)
            bm = bm.repeat_interleave(self.num_flip_combinations, dim = 0)
            # bm = bm.repeat((self.num_flip_combinations, 1))
        else:
            z = torch.cat((noise,bm), dim=1)
            levy = self.netG(z)

        return torch.cat((bm, levy), dim=1)
