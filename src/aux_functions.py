import math
import timeit
from timeit import default_timer

import numpy
import torch
import numpy as np
import ot
from math import sqrt
from torch.distributions import Bernoulli


def chen_combine(bm_levy_in: torch.Tensor, _bm_dim: int, uniform_s: bool = False, chunking: bool = False):
    """Given two samples and bm and their levy areas, performs Chen's identity: A_[0,1]^ij = A_[0,1/2]^ij + A_[1/2,1]^ij + (1/2)*(W_[0,1/2]^iW_[1/2,1]^j - W_[0,1/2]^jW_[1/2,1]^i). Input increments and levy areas are assumed to be over [0, 1], so Brownian rescaling is applied so inputs are over [0,1/2].

    Args:
        bm_levy_in (torch.Tensor): input increments of bm and levy areas
        _bm_dim (int): dimension of bm

    Returns:
        torch.Tensor: sample of brownian motion and levy area after one iteration of Chen's identity. Note the length of the output is half the input length.
    """
    _levy_dim = int((_bm_dim * (_bm_dim - 1)) // 2)
    device = bm_levy_in.device
    # the batch dimension of the inputs will be quartered
    out_size = bm_levy_in.shape[0] // 2
    assert 2 * out_size == bm_levy_in.shape[0]
    assert bm_levy_in.shape[1] == _bm_dim + _levy_dim

    start_time = default_timer()
    bm_strided = bm_levy_in.view(out_size, 2, bm_levy_in.shape[1])
    if chunking:
        wl_0_s, wl_s_t = bm_levy_in.chunk(2)
    else:
        wl_0_s = bm_strided[:, 0]
        wl_s_t = bm_strided[:, 1]

    if uniform_s:
        s = torch.rand(wl_0_s.shape[0], 1).to(wl_0_s.device)
        # w_0_s is from 0 to s and wl_s_t is from s to 1
        w_0_s = torch.sqrt(s) * wl_0_s[:, :_bm_dim]
        w_s_t = torch.sqrt(1 - s) * wl_s_t[:, :_bm_dim]

        l_0_s = s * wl_0_s[:, _bm_dim:]
        l_s_t = (1 - s) * wl_s_t[:, _bm_dim:]

        new_wl_0s = torch.cat([w_0_s, l_0_s], dim=-1)
        new_wl_s_t = torch.cat([w_s_t, l_s_t], dim=-1)
        result = torch.clone(new_wl_0s + new_wl_s_t)
        idx = _bm_dim
        for k in range(_bm_dim - 1):
            for l in range(k + 1, _bm_dim):
                correction_term = 0.5 * torch.sqrt(s).squeeze(1) * torch.sqrt(1 - s).squeeze(1) * (
                        wl_0_s[:, k] * wl_s_t[:, l]
                        - wl_0_s[:, l] * wl_s_t[:, k])
                result[:, idx] += correction_term
                idx += 1
    else:

        result = torch.clone(wl_0_s + wl_s_t)

        # Perform rescaling
        result[:, :_bm_dim] *= sqrt(0.5)
        result[:, _bm_dim:(_bm_dim + _levy_dim)] *= 0.5

        # This way is just as fast
        w_0_s = wl_0_s[:, :_bm_dim]
        w_s_t = wl_s_t[:, :_bm_dim]
        correction_terms = w_0_s.view(-1, _bm_dim, 1) * w_s_t.view(-1, 1, _bm_dim) - \
                           w_0_s.view(-1, 1, _bm_dim) * w_s_t.view(-1, _bm_dim, 1)

        triu_indices = torch.triu_indices(_bm_dim, _bm_dim, offset=1, device=device)
        result[:, _bm_dim:] += 0.25 * correction_terms[:, triu_indices[0], triu_indices[1]]

    return result


def MC_chen_combine(bm_levy_in_: np.array, bm_dim: int):
    """Takes in samples in shape for MC simulation. Reshapes, performs Chen combine pairwise then reshapes back to MC shape.

    Args:
        bm_levy_in_ (np.array): input bm and levy area
        bm_dim (int): dimensions of brownian motion
    """
    # Reshape to perform chen combine
    M, N, _ = bm_levy_in_.shape
    bm_levy_in = bm_levy_in_.view(M * N, -1)
    bm_levy_out = chen_combine(bm_levy_in, bm_dim)
    # Undo scaling from chen_combine
    bm_levy_out[:, :bm_dim] *= sqrt(2)
    bm_levy_out[:, bm_dim:] *= 2
    # Reshape back to MC shape
    bm_levy_out_ = bm_levy_out.view(M, N // 2, -1)
    return bm_levy_out_


def levy_conditional_st_dev(_bm: list, T=1):
    """Computes the the conditional standard deviation of levy area give bm

    Args:
        _bm (list): increment of bm
        T (optional): step size. Defaults to 1.

    Returns:
        _type_: sqrt{E[(A^ij)^2 | W]}
    """
    # Convert list to numpy array
    _bm = np.array(_bm)
    _bm_dim = _bm.shape[0]
    _bm_squared = np.square(_bm)
    _levy_dim = int((_bm_dim - 1) * _bm_dim / 2)
    st_devs = []
    for k in range(_bm_dim):
        for l in range(k + 1, _bm_dim):
            st_devs.append(sqrt((1.0 / 12.0) * (1.0 * T ** 2 + T * (_bm_squared[k] + _bm_squared[l]))))
    assert len(st_devs) == _levy_dim
    return st_devs


def empirical_second_moments(_levy_generated: np.ndarray):
    """Computes the empirical joint second moments of generated samples of Levy areas

    Args:
        _levy_generated (np.ndarray): input sample of levy areas

    Returns:
        np.ndarray: matrix with i,j^th component the empirical second moment between the i^th and j^th coordinate of the Levy area input array
    """
    _batch_dim = _levy_generated.shape[0]
    const = 1.0 / _batch_dim
    _levy_dim = _levy_generated.shape[1]
    result = np.zeros((_levy_dim, _levy_dim))
    for i in range(_levy_dim):
        for j in range(i, _levy_dim):
            result[i, j] = const * np.dot(_levy_generated[:, i], _levy_generated[:, j])
            if i != j:
                result[j, i] = result[i, j]

    return result


def joint_wass_dist(x1: torch.Tensor, x2: torch.Tensor, numItermax=1e06):
    """Returns the joint Wasserstein2 distance between two samples

    Args:
        x1 (torch.Tensor): first sample
        x2 (torch.Tensor): second sample

    Returns:
        torch.tensor: joint Wasserstein2 distance
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    closest_pairing_matrix = ot.dist(x1, x2, metric='sqeuclidean')
    a = torch.tensor([], dtype=torch.float, device=device)
    b = torch.tensor([], dtype=torch.float, device=device)
    return torch.sqrt(ot.emd2(a, b, closest_pairing_matrix, numItermax=numItermax))


def bm_indices(levy_i: int, _bm_dim: int):
    """Given index of levy area, return the correpsonding indices of bm

    Args:
        levy_i (int): index of levy area
        _bm_dim (int): dimension of bm

    Returns:
        int, int: indices of bm
    """
    if levy_i >= int(_bm_dim * (_bm_dim - 1) // 2) or levy_i < 0:
        return None
    idx = 0
    for k in range(_bm_dim):
        for l in range(k + 1, _bm_dim):
            if idx == levy_i:
                return k, l
            else:
                idx += 1


def list_pairs(_bm_dim: int, _bm=None):
    """Given the increment of bm, returns all possible pairs of coordinates

    Args:
        _bm_dim (int): dimension of brownian motion
        _bm (list, optional): increment of brownian motion. Defaults to None.

    Returns:
        list of tuples: list of pairs of increments
    """
    fixed_bm_list = [1.0, -0.5, -1.2, -0.3, 0.7, 0.2, -0.9, 0.1, 1.7]
    if not (_bm is None):
        fixed_bm_list = list(_bm)
        assert (len(fixed_bm_list) == _bm_dim)
    lst = []
    for k in range(_bm_dim):
        for l in range(k + 1, _bm_dim):
            lst.append((fixed_bm_list[k], fixed_bm_list[l]))
    return lst


# All pairs of increments for bm between dimension 0 and 9
pair_lists = [list_pairs(wd) for wd in range(10)]


def mom4_gpu(bm_dim: int, batch_size: int, bm_in: torch.Tensor = None, k_in: torch.Tensor = None,
             h_in: torch.Tensor = None, device_to_use: torch.device = None):
    """ GPU implementation of Fosters weak method for simulating levy area, matching first four moments

    Args:
        bm_dim (int): dimension of bm
        batch_size (int): batch size
        bm_in (torch.Tensor, optional): input batch of brownian motion. If None, then generates a sample. Defaults to None.
        k_in (torch.Tensor, optional): input batch of space-space-time levy areas. If None, then generates a sample. Defaults to None.
        h_in (torch.Tensor, optional): input batch of space-time levy areas. If None, then generates a sample. Defaults to None.

    Returns:
        torch.Tensor: bsz x bm inrements x levy areas
    """

    # SETUP
    levy_dim = int(bm_dim * (bm_dim - 1) // 2)
    start = torch.cuda.Event(enable_timing=True)
    square_K_event = torch.cuda.Event(enable_timing=True)
    gen_C = torch.cuda.Event(enable_timing=True)
    gen_ber = torch.cuda.Event(enable_timing=True)
    gen_uni = torch.cuda.Event(enable_timing=True)
    gen_rad = torch.cuda.Event(enable_timing=True)

    sigma_event = torch.cuda.Event(enable_timing=True)
    sigma2_event = torch.cuda.Event(enable_timing=True)

    bmh_event = torch.cuda.Event(enable_timing=True)
    kh_event = torch.cuda.Event(enable_timing=True)

    index_select_event = torch.cuda.Event(enable_timing=True)

    out_cat_event = torch.cuda.Event(enable_timing=True)

    if device_to_use is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = device_to_use

    if bm_in is None:
        bm = torch.randn(size=(batch_size, bm_dim), dtype=torch.float, device=device)
    else:
        bm = bm_in
    if h_in is None:
        h = sqrt(1 / 12) * torch.randn(size=(batch_size, bm_dim), dtype=torch.float, device=device)
    else:
        h = h_in
    if k_in is None:
        k = sqrt(1 / 720) * torch.randn(size=(batch_size, bm_dim), dtype=torch.float, device=device)
    else:
        k = k_in

    # construct ingredients
    start.record()

    squared_K = torch.square(k)
    square_K_event.record()
    C = torch.empty(size=(batch_size, bm_dim), dtype=torch.float, device=device)
    C.exponential_(15 / 8)
    gen_C.record()
    p = 21130 / 25621
    p = torch.tensor(p, dtype=torch.float, device=device)
    c = sqrt(1 / 3) - 8 / 15
    bernoulli_distn = torch.distributions.Bernoulli(probs=p)
    ber = bernoulli_distn.sample(sample_shape=(batch_size, levy_dim))
    gen_ber.record()
    uni = torch.empty(size=(batch_size, levy_dim), dtype=torch.float, device=device)
    uni.uniform_(-sqrt(3), sqrt(3))
    gen_uni.record()
    half = torch.tensor(0.5, dtype=torch.float, device=device)
    bernoulli_half = torch.distributions.Bernoulli(probs=half)
    rademacher = 1 - 2 * bernoulli_half.sample(sample_shape=(batch_size, levy_dim))
    ksi = ber * uni + (1 - ber) * rademacher
    gen_rad.record()

    if bm_dim >= 6:
        # ========= New approach =========
        C_plus_c: torch.Tensor = C + c
        C_plus_c_trsp = (3 / 28) * C_plus_c.unsqueeze(1)
        C_plus_c = C_plus_c.unsqueeze(2)
        sigma_matrix = torch.bmm(C_plus_c, C_plus_c_trsp)
        sigma_event.record()
        squared_K = (144 / 28) * squared_K
        sigma_matrix = sigma_matrix + squared_K.unsqueeze(1) + squared_K.unsqueeze(2)
        sigma_matrix = torch.sqrt(sigma_matrix)
        sigma2_event.record()

        h_unsqeezed = h.unsqueeze(1)

        bmh = torch.bmm(bm.unsqueeze(2), h_unsqeezed)
        bmh = (bmh.permute(0, 2, 1) - bmh)
        bmh_event.record()

        kh = torch.bmm(k.unsqueeze(2), h_unsqeezed)
        kh = kh - kh.permute(0, 2, 1)
        bmh = bmh + 12 * kh
        kh_event.record()

        indices = torch.triu_indices(bm_dim, bm_dim, offset=1, device=device)
        bmh = bmh[:, indices[0], indices[1]]
        tilde_a = ksi * (sigma_matrix[:, indices[0], indices[1]])
        index_select_event.record()

        levy_out = tilde_a + bmh
        out = torch.cat((bm, levy_out), dim=1)

    else:
        # ========== Old approach ==========
        # create the sigma RV
        def sigma(i: int, j: int):
            return torch.sqrt(3 / 28 * (C[:, i] + c) * (C[:, j] + c) + 144 / 28 * (squared_K[:, i] + squared_K[:, j]))

        idx = 0
        for j in range(bm_dim):
            for l in range(j + 1, bm_dim):
                sig = sigma(j, l)
                # now calculate a from ksi and sigma (but store a in ksi)
                ksi[:, idx] *= sig
                # calculate the whole thing
                ksi[:, idx] += h[:, j] * bm[:, l] - bm[:, j] * h[:, l] + 12 * (
                        k[:, j] * h[:, l] - h[:, j] * k[:, l])
                idx += 1
        levy_out = ksi
        out = torch.cat((bm, levy_out), dim=1)

    out_cat_event.record()
    torch.cuda.synchronize()
    return out


def Davie_gpu(bm_in: torch.Tensor, device, h_in: torch.Tensor = None, bb: torch.Tensor = None):
    """
    Performs bridge flipping but directly with Rad RVs (not with all sign combinations)
    Args:
        bm_in (torch.Tensor):
        bb_in (torch.Tensor):
        h_in (torch.Tensor):
        device (torch.device):
        r_in (torch.Tensor): (Optional) Rademacher random variables

    Returns:
        torch.Tensor: Levy area
    """
    bm_dim = bm_in.shape[1]
    levy_dim = int(bm_dim * (bm_dim - 1) // 2)
    bsz = bm_in.shape[0]

    if h_in is None:
        h_in = sqrt(1 / 12) * torch.randn((bsz, bm_dim), dtype=torch.float, device=device)

    H = h_in.unsqueeze(1)
    BM = bm_in.unsqueeze(2)

    BMH = torch.bmm(BM, H)
    BMH = (BMH.permute(0, 2, 1) - BMH)
    indices = torch.triu_indices(bm_dim, bm_dim, offset=1, device=device)
    BMH = BMH[:, indices[0], indices[1]]

    if bb is None:
        bb = sqrt(1 / 12) * torch.randn((bsz, levy_dim), dtype=torch.float, device=device)

    return BMH + bb


def Davie_gpu_all(_bm_dim: int, _batch_size: int):
    """
    Performs bridge flipping but directly with Rad RVs (not with all sign combinations)
    Args:
        bm_in (torch.Tensor):
        bb_in (torch.Tensor):
        h_in (torch.Tensor):
        device (torch.device):
        r_in (torch.Tensor): (Optional) Rademacher random variables

    Returns:
        torch.Tensor: (bm_in, Levy area)
    """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    bm_in = torch.randn((_batch_size, _bm_dim), dtype=torch.float, device=device)
    bm_dim = bm_in.shape[1]
    levy_dim = int(bm_dim * (bm_dim - 1) // 2)
    bsz = bm_in.shape[0]

    h_in = sqrt(1 / 12) * torch.randn((bsz, bm_dim), dtype=torch.float, device=device)

    H = h_in.unsqueeze(1)
    BM = bm_in.unsqueeze(2)

    BMH = torch.bmm(BM, H)
    BMH = (BMH.permute(0, 2, 1) - BMH)
    indices = torch.triu_indices(bm_dim, bm_dim, offset=1, device=device)
    BMH = BMH[:, indices[0], indices[1]]

    bb = sqrt(1 / 12) * torch.randn((bsz, levy_dim), dtype=torch.float, device=device)

    return torch.cat([bm_in, BMH + bb], dim=1)


def rademacher_GPU_dim2(_batch_size: int):
    """Returns brownian increments and rademacher random variables matching the conditional variance of levy area given the brownian increment.

    Args:
        _batch_size (int): number of samples

    Returns:
        torch.tensor: brownian increments and levy areas
    """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    bm_in = torch.randn((_batch_size, 2), dtype=torch.float, device=device)
    std = torch.sqrt(1 / 12 + (1 / 12) * (bm_in[:, 0] ** 2 + bm_in[:, 1] ** 2)).view(_batch_size, -1)
    area_ = 2 * (Bernoulli(torch.tensor([0.5])).sample((_batch_size,)) - 0.5).to(device)
    area = std * area_

    return torch.cat([bm_in, area], dim=1)


def rademacher_GPU_dim2_var(_batch_size: int):
    """Returns brownian increments and independent rademacher random variables matching the variance of levy area

    Returns:
        torch.tensor: brownian increments and levy areas
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    bm_in = torch.randn((_batch_size, 2), dtype=torch.float, device=device)
    std = torch.tensor(1 / 2).to(device)
    area_ = 2 * (Bernoulli(torch.tensor([0.5])).sample((_batch_size,)) - 0.5).to(device)
    area = std * area_

    return torch.cat([bm_in, area], dim=1)


def fourth_moments(input_samples: np.ndarray):
    """Empirical fourth moments (including cross moments) of a sample

    Args:
        input_samples (np.ndarray | torch.Tensor): sample

    Returns:
        _type_: empirical fourth moments
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if isinstance(input_samples, np.ndarray):
        input_samples = torch.tensor(input_samples, dtype=torch.float, device=device)
    dim = input_samples.shape[1]
    res = []
    for i in range(dim):
        for j in range(i, dim):
            ij = input_samples[:, i] * input_samples[:, j]
            for k in range(j, dim):
                ijk = ij * input_samples[:, k]
                for l in range(k, dim):
                    col = ijk * input_samples[:, l]
                    res.append(torch.mean(col, dim=0, keepdim=True))
    out = torch.cat(res, dim=0)
    return out


def fourth_moment_errors(input_samples: np.ndarray, true_moments: torch.Tensor,
                         loss_type: str = 'max'):
    """
    Compute the empirical fourth moments of fake_samples and return the mean absolute difference
    between those and the desired fourth moments
    Args:
        input_samples:
        true_moments:
        loss_type (str): 'mean_abs' or 'max' or 'RMS'
    Returns:

    """
    fake_moments = fourth_moments(input_samples)
    diff = true_moments - fake_moments
    if loss_type == 'mean_abs':
        out = torch.abs(diff).mean()
    elif loss_type == 'RMS':
        out = torch.sqrt(torch.mean(torch.pow(diff, exponent=2)))
    elif loss_type == 'max':
        out = torch.max(torch.abs(diff))
    else:
        raise ValueError("loss type should be 'mean_abs' or 'max' or 'RMS'")
    return out


def make_pretty(errs, decimal_places=5):
    """Rounds input to fixed number of decimal places

    Args:
        errs : array to be rounded
        decimal_places (int, optional): decimal places to round to. Defaults to 5.

    Returns:
        : rounded input
    """
    if isinstance(errs, list):
        if len(errs) == 1:
            return float(f"{errs[0]:.{decimal_places}f}")
        return [float(f"{i:.{decimal_places}f}") for i in errs]
    if isinstance(errs, float):
        return float(f"{errs:.{decimal_places}f}")
    else:
        return errs


def read_serial_number(dict_saves_folder):
    """Looks at all models with same configuration and reads back least serial number not yet used. To ensure multiple models with same configuration do not overwrite each other.
    Args:
        dict_saves_folder (str): path to this model's directory in model_saves/
    Returns:
        int: least serial number not yet used for model config
    """
    filename = f'model_saves/{dict_saves_folder}/summary_file.txt'
    with open(filename, 'a+') as summary_file:
        summary_file.seek(0)
        lines = summary_file.read().splitlines()
        if not lines:
            serial_num = 1
            summary_file.write(f"0 SUMMARY FILE FOR: {dict_saves_folder}\n")
        else:
            last_line = lines[-1]
            serial_num = int(last_line.split()[0]) + 1
    return serial_num


def fast_flipping(bm_in: torch.Tensor, bb_in: torch.Tensor, h_in: torch.Tensor, device, r_in=None, T_in=None):
    """
    Performs bridge flipping but directly with Rad RVs (not with all sign combinations)
    Args:
        bm_in (torch.Tensor):
        bb_in (torch.Tensor):
        h_in (torch.Tensor):
        device (torch.device):
        r_in (torch.Tensor): (Optional) Rademacher random variables

    Returns:
        torch.Tensor: Levy area
    """
    bm_dim = bm_in.shape[1]
    bsz = bm_in.shape[0]
    if r_in is None:
        ber_distn = Bernoulli(probs=torch.tensor([0.5], dtype=torch.float, device=device))
        r = 2 * (ber_distn.sample((bsz, bm_dim)).to(device) - 0.5).squeeze()
    else:
        r = r_in

    H = (h_in * r).unsqueeze(1)
    BM = bm_in.unsqueeze(2)

    BMH = torch.bmm(BM, H)
    BMH = (BMH.permute(0, 2, 1) - BMH)
    indices = torch.triu_indices(bm_dim, bm_dim, offset=1, device=device)
    BMH = BMH[:, indices[0], indices[1]]

    M = torch.bmm(r.unsqueeze(2), r.unsqueeze(1))[:, indices[0], indices[1]]
    Mbb = M * bb_in

    return BMH + Mbb


def poly_expansion_bb(h_in: torch.Tensor, num_terms: int = 2, precision: float = None, max_mem_alloc=512,
                      gen_bb=True, bm_in=None):
    """
    Use Foster's polynomial expansion to generate the space-space Levy area of the Brownian bridge
     conditional on H=h_in.
    Args:
        h_in:
        num_terms: The numer of terms to use
        precision: The desired precision (if supplied overrides num_terms)
        max_mem_alloc: the maximal memory allocated to this function (in MB)
        gen_bb: if false will generate Levy area A

    Returns:

    """
    bsz = h_in.shape[0]
    bm_dim = h_in.shape[1]

    if precision is not None:
        num_terms = max(math.ceil(1 / (8 * pow(precision, exp=2))), 2)  # this is not exact, but that's fine

    res = torch.zeros((bsz, bm_dim, bm_dim), dtype=torch.float, device=h_in.device)

    # to avoid a long for loop compute in chunks
    chunk_len = max(int(max_mem_alloc * (2 ** 19) / (bsz * bm_dim * bm_dim)), 4)

    num_chunks = math.ceil(num_terms / chunk_len)

    # this is the overlap term between chunks with k = (2 * i * chunk_len + 1)
    prev_c = 2 * h_in

    even_ks = torch.arange(start=2, end=2 * chunk_len + 2, step=2, dtype=torch.float, device=h_in.device)
    odd_ks = torch.arange(start=3, end=2 * chunk_len + 1, step=2, dtype=torch.float, device=h_in.device)

    for i in range(num_chunks):
        if i == num_chunks - 1 and num_terms % chunk_len > 0:  # last chunk is shorter
            chunk_len = num_terms % chunk_len
            even_ks = even_ks[:chunk_len]
            odd_ks = odd_ks[:chunk_len - 1]  # this one is shorter
        even_cs = torch.randn((chunk_len, bsz, bm_dim), dtype=torch.float, device=h_in.device)
        even_cs = even_cs * torch.pow(1 + 2 * even_ks.view(-1, 1, 1), exponent=-0.5)

        odd_cs = torch.randn((chunk_len - 1, bsz, bm_dim), dtype=torch.float, device=h_in.device)
        odd_cs = odd_cs * torch.pow(1 + 2 * odd_ks.view(-1, 1, 1), exponent=-0.5)

        # last odd c
        last_k = 2 * (i + 1) * chunk_len + 1
        last_c = pow(2 * last_k + 1, exp=-0.5) * torch.randn((bsz, bm_dim), dtype=torch.float, device=h_in.device)

        # c_k^i c_{k+1}^j where k is odd
        first_term_odd_even = prev_c.unsqueeze(2) * even_cs[0].unsqueeze(1)
        odd_even = odd_cs.unsqueeze(3) * even_cs[1:].unsqueeze(2)
        odd_even = first_term_odd_even + torch.sum(odd_even, dim=0, keepdim=False)

        # c_k^i c_{k+1}^j where k is even
        last_term_even_odd = even_cs[-1].unsqueeze(2) * last_c.unsqueeze(1)
        even_odd = even_cs[:-1].unsqueeze(3) * odd_cs.unsqueeze(2)
        even_odd = last_term_even_odd + torch.sum(even_odd, dim=0, keepdim=False)

        # print(f"After even_odd: {torch.cuda.memory_allocated()//1024}")

        res += odd_even + even_odd

        even_ks = 2 * chunk_len + even_ks
        odd_ks = 2 * chunk_len + odd_ks

        prev_c = last_c

    # at this point res = \sum c_k^i c_{k+1}^j, so we need the antisym term

    res = res - res.permute(0, 2, 1)

    triu_indices = torch.triu_indices(row=bm_dim, col=bm_dim, offset=1, device=h_in.device)
    res = 0.5 * res[:, triu_indices[0], triu_indices[1]]

    if not gen_bb:  # then use Davie_gpu to compute Levy area
        if bm_in is None:
            bm_in = torch.randn((bsz, bm_dim), dtype=torch.float, device=h_in.device)

        res = Davie_gpu(bm_in, device=h_in.device, h_in=h_in, bb=res)

    return res
