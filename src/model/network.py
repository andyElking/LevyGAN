import torch.nn as nn
import torch


class PairNetBatchNorm(nn.Module):
    def __init__(self, num_fts):
        super().__init__()
        self.eps = 1e-05
        self.num_fts = num_fts
        self.gamma = nn.Parameter(
            torch.ones((1, 1, num_fts), dtype=torch.float), requires_grad=True
        )
        self.beta = nn.Parameter(
            torch.zeros((1, 1, num_fts), dtype=torch.float), requires_grad=True
        )

    def forward(self, x, levy_dim):
        bsz = x.shape[0] // levy_dim
        assert x.shape == (bsz * levy_dim, self.num_fts)
        x_expanded: torch.Tensor = x.view(bsz, levy_dim, self.num_fts)
        batch_avg = x_expanded.mean(dim=0, keepdim=True)
        batch_var_inv = torch.pow(
            torch.var(x_expanded, dim=0, keepdim=True) + self.eps, exponent=-1
        )
        x_normalised = (x_expanded - batch_avg) * (
            batch_var_inv * self.gamma
        ) + self.beta
        return x_normalised.view(bsz * levy_dim, self.num_fts).contiguous()


class LevyGANLayer(nn.Module):
    def __init__(
        self, layer_indim, layer_outdim, use_batch_norm, act_type, use_pairnet_bn
    ):
        super(LevyGANLayer, self).__init__()
        self.act = None
        if act_type is not None:
            self.act = get_act(act_type, layer_outdim)
        self.net = nn.Linear(layer_indim, layer_outdim, bias=(not use_batch_norm))
        self.batch_norm = None
        if use_batch_norm:
            self.batch_norm = (
                PairNetBatchNorm(layer_outdim)
                if use_pairnet_bn
                else nn.BatchNorm1d(layer_outdim)
            )

    def forward(self, x, levy_dim=None):
        x = self.net(x)
        if self.batch_norm is not None:
            if isinstance(self.batch_norm, PairNetBatchNorm):
                x = self.batch_norm.forward(x, levy_dim)
            else:
                x = self.batch_norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


def get_act(act_type, num_fts):
    if act_type == "relu":
        return nn.ReLU()
    elif isinstance(act_type, tuple) and act_type[0] == "leaky":
        slope = act_type[1]
        assert isinstance(slope, float) and 0 <= slope < 1
        return nn.LeakyReLU(slope)
    elif "none":
        return None


def create_net(gen_conf: dict, in_dim: int, out_dim: int):
    """
    Creates the neural net for the generator
    Args:
        gen_conf:
        in_dim: dimensionality of inputs (excluding batch dimensions)
        out_dim: dimensionality of outputs (excluding batch dimensions)
    Returns:
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    bm_dim = gen_conf["bm_dim"]
    levy_dim = int((bm_dim * (bm_dim - 1)) // 2)
    hidden_dim = gen_conf["hidden_dim"] if "hidden_dim" in gen_conf else 64
    num_layers = gen_conf["num_layers"] if "num_layers" in gen_conf else 3
    activations = (
        gen_conf["activation"]
        if "activation" in gen_conf
        else [("leaky", 0.2)] * (num_layers - 1)
    )
    batch_norm = gen_conf["batch_norm"] if "batch_norm" in gen_conf else False

    pairnet_bn = gen_conf["pairnet_bn"] if "pairnet_bn" in gen_conf else False

    if not (isinstance(activations, list)):
        activations = [activations] * (num_layers - 1)
    assert len(activations) >= num_layers - 1

    # ===== Describe the neural net =======

    def activation_description(act_type):
        if isinstance(act_type, tuple) and act_type[0] == "leaky":
            return f"lky{act_type[1]}"
        return str(act_type)

    act_desc = ""
    for act_type in activations:
        act_desc += activation_description(act_type)

    net_description = (
        f"{num_layers}LAY_{hidden_dim}HID_{act_desc}ACT{'_BN' if batch_norm else ''}"
    )

    # ======== Assemble the neural net ======

    assert num_layers >= 1

    def get_layer(layer_num):
        """
        Args:
            layer_num: which layer in sequence this is
        Returns:
            list[nn.Module]
        """
        if layer_num == 0:
            layer_indim = in_dim
        else:
            layer_indim = hidden_dim

        if layer_num == num_layers - 1:
            return LevyGANLayer(layer_indim, out_dim, False, None, pairnet_bn)
        else:
            layer_outdim = hidden_dim

        layer = LevyGANLayer(
            layer_indim, layer_outdim, batch_norm, activations[layer_num], pairnet_bn
        )
        return layer

    layers = nn.ModuleList()
    for i in range(num_layers):
        layers.append(get_layer(i))

    layers = layers.to(device)
    return layers, net_description
