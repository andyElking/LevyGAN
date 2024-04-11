"""
This file should be used to describe the whole evaluation procedure, during and after the training.

Todo: We should implement a function full_evaluation() that takes data_in a generator and a config file, so we do not need to load the trainer everytime.

Todo: Also, we need to factorize the set of test metric from the evaluation module

"""
from statistics import mean

import torch
import numpy as np
from src.model.Generator import Generator
from torch.autograd import grad as torch_grad
from src.aux_functions import *
import matplotlib.pyplot as plt
from timeit import default_timer
from os import listdir
from os.path import isfile, join


def compute_grad_norm(generator, discriminator, sample_data, config, device):
    """
    This function computes the gradient norm of the discriminator and the penalty coefficient
    :param generator:
    :param sample_data:
    :param config:
    :return:
    """
    unfixed_test_bsz = config["unfixed_test_bsz"]
    unfixed_data = sample_data[:unfixed_test_bsz]
    actual_bsz = unfixed_data.shape[0]

    bm = unfixed_data[:, : config["bm_dim"]]
    fake_data = generator(bm).detach()
    gradient_penalty, gradient_norm = _gradient_penalty(
        unfixed_data, fake_data, gp_weight=0, discriminator=discriminator, device=device
    )

    return gradient_norm, gradient_penalty


def _gradient_penalty(real_data, generated_data, gp_weight, discriminator, device):
    """
    Computes the gradient penalty coefficients used during the training of discriminator
    :param real_data:
    :param generated_data:
    :param gp_weight:
    :param discriminator:
    :param device:
    :return:
    """
    b_size_gp = real_data.shape[0]
    # Calculate interpolation
    alpha = torch.rand(b_size_gp, 1, device=device)
    alpha = alpha.expand_as(real_data)
    interpolated = (alpha * real_data + (1 - alpha) * generated_data).requires_grad_(
        True
    )

    interpolated = interpolated.to(device)

    # Calculate probability of interpolated examples
    prob_interpolated = discriminator(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones(prob_interpolated.size(), device=device),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Gradients have shape (b_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(b_size_gp, -1)
    # grad_norm = gradients.norm(2, dim=1).mean().item()

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)
    avg_grad_norm = gradients_norm.mean().item()

    # Return gradient penalty
    return gp_weight * ((gradients_norm - 1) ** 2).mean(), avg_grad_norm


def compute_fixed_losses(
    generator,
    config,
    input_true_data: torch.Tensor,
    input_true_bm: torch.Tensor,
    true_fourth_moments=None,
    comp_joint_error=False,
):
    """
    Compute various losses for model evaluation:
        1) avg_st_dev_error: the second moment difference between real and fake Levy terms
        2) joint_w2_loss: Wasserstein loss of the joint process (L_ij)
        3) marginal_w2_loss: Wassertein loss computed at each coordinate

    :param generator:
    :param config:
    :param input_true_data:
    :param device:
    :param comp_joint_error:
    :return:
    """

    # prepare data_ins
    test_bsz = input_true_data.shape[0]

    bm_dim = input_true_bm.shape[0]
    levy_dim = int((bm_dim * (bm_dim - 1)) // 2)
    if input_true_data.shape[1] != levy_dim:
        print(
            f"Warning in compute_fixed_losses: input_true_bm.shape={input_true_bm.shape}, but "
            f"input_true_data.shape={input_true_data.shape}"
        )
    if len(input_true_bm.shape) == 1:
        bm = input_true_bm.expand(test_bsz, bm_dim)
    else:
        bm = input_true_bm

    levy_true = input_true_data[:test_bsz]

    # compute correct st_dev
    st_dev_fixed_increment = np.diag(
        levy_conditional_st_dev(bm[1, :].to("cpu").tolist())
    )

    # run generator

    fake_data = generator.sample_fake_data(bm)
    levy_fake = fake_data[:, bm_dim : (bm_dim + levy_dim)]
    marginal_w2_loss = [
        sqrt(ot.wasserstein_1d(levy_true[:, i], levy_fake[:, i], p=2).item())
        for i in range(levy_dim)
    ]

    if bm_dim >= 3:
        levy_fake_np = levy_fake.to("cpu").numpy()
        st_dev_err = avg_st_dev_error(st_dev_fixed_increment, levy_fake_np)
        # self.test_results['st dev error'] = st_dev_err
    else:
        st_dev_err = float("inf")

    if comp_joint_error and (bm_dim >= 3):
        joint_wass_dist_bsz = config["joint_wass_dist_bsz"]
        joint_w2_loss = joint_wass_dist(
            levy_true[:joint_wass_dist_bsz], levy_fake[:joint_wass_dist_bsz]
        )
    else:
        joint_w2_loss = float("inf")

    if true_fourth_moments is not None:
        mom4_errs = fourth_moment_errors(
            input_samples=levy_fake, true_moments=true_fourth_moments
        ).item()

    return marginal_w2_loss, joint_w2_loss, st_dev_err, mom4_errs


def compute_joint_losses(generator, config, input_true_data, device, bm_dim=None):
    """
    Computes the joint law of joint Brownian Motion and Levy area
    :param generator:
    :param config:
    :param input_true_data:
    :param device:
    :return:
    """
    bm_dim = generator.bm_dim if bm_dim is None else bm_dim
    bm_dim = generator.bm_dim
    levy_dim = int((bm_dim * (bm_dim - 1)) // 2)

    joint_wass_dist_bsz = config["joint_wass_dist_bsz"]

    # Randomly choose samples from real data
    indices = torch.randperm(input_true_data.shape[0])[:joint_wass_dist_bsz]
    true_data = input_true_data[indices]
    true_data_np = true_data.to("cpu").numpy()
    bm_torch = true_data[:, :bm_dim]
    levy_true = true_data[:, bm_dim : (bm_dim + levy_dim)]

    # run generator
    fake_data = sample_fake_data(generator, bm_torch)

    levy_fake = fake_data[:, bm_dim : (bm_dim + levy_dim)]

    joint_w2_loss = joint_wass_dist(levy_true, levy_fake)

    return joint_w2_loss


def avg_st_dev_error(st_dev_fixed_increment, _levy_generated):
    difference = np.abs(
        st_dev_fixed_increment
        - np.sqrt(np.abs(empirical_second_moments(_levy_generated)))
    )
    return difference.mean()


def make_report(
    tester, epoch: int = None, iters: int = None, add_line_break=True, short=False
):
    """
    Generate a string that contains all the test metrics
    :param tester: The trainer object
    :param epoch:
    :param iters:
    :param chen_iters:
    :param add_line_break:
    :param short:
    :return:
    """
    test_results = tester.test_results

    report = ""
    if add_line_break:
        line_break = "\n"
    else:
        line_break = ", "

    if iters is not None:
        report += f"itr: {iters}, "

    score = test_results["best score"]
    best_4mom = test_results["best_max_4mom_errors"]
    report += f"best_scr: {score:.3f}, 100best_4mom: {best_4mom*100:.3f} "
    # grad_norm = test_results['gradient norm']
    # report += f"discr grad norm: {grad_norm:.5f}, "
    if len(test_results["loss d"]) > 0:
        if short:
            avg_loss_d = mean(test_results["loss d"])
            report += f"discr loss: {make_pretty(avg_loss_d)}"
        else:
            report += f"discr loss: {make_pretty(test_results['loss d'])}"

    joint_wass_errors = test_results["joint wass errors"]
    if len(joint_wass_errors) > 0:
        if short:
            avg_joint = mean(joint_wass_errors)
            report += f", averaged joint w1 loss: {make_pretty(avg_joint)}"
        else:
            report += f", joint w1 loss: {make_pretty(joint_wass_errors)}"

    st_dev_error = test_results["st dev error"]
    if st_dev_error < 100 and not short:
        report += f", st dev loss: {st_dev_error:.5f}"

    if len(test_results["marginal w2 loss"]) > 0:
        if short:
            avg_err = mean(test_results["marginal w2 loss"])
            report += f" avg margW2 loss: {make_pretty(avg_err)}"
        else:
            report += f"{line_break}marginal w2 loss: {make_pretty(test_results['marginal w2 loss'])}"

    if len(test_results["fourth_mom_errors"]) > 0:
        if short:
            avg_err = mean(test_results["fourth_mom_errors"])
            report += f" 4mom loss: {make_pretty(avg_err)}"
        else:
            report += f"{line_break} max 4th mom errors: {make_pretty(test_results['fourth_mom_errors'])}"

    return report


def draw_error_graphs(
    graph_filename,
    marginal_w2_losses_through_training,
    single_coord_labels,
    discriminative_losses_through_training,
    fourth_moments_through_training=None,
    joint_labels=None,
    descriptor: str = "",
    iters=None,
):
    """
    Plot the losses with respect to training iterations
    :param graph_filename:
    :param marginal_w2_losses_through_training:
    :param single_coord_labels:
    :param discriminative_losses_through_training:
    :param fourth_moments_through_training:
    :param joint_labels:
    :param descriptor:
    :return:
    """
    plt.rcParams.update({"font.size": 15})
    if (fourth_moments_through_training is not None) and len(
        fourth_moments_through_training
    ) > 0:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(28, 7))
        ax3.set_title("Maximal error in 4th moments")
        ax3.plot(iters, fourth_moments_through_training, label=joint_labels)
        ax3.set_ylim([0, 0.1])
        ax3.set_xlabel("iterations")
        ax3.legend(prop={"size": 10})
        ax2.set_xlabel("iterations")
        ax2.set_title("Discriminator losses")
        ax2.plot(iters, discriminative_losses_through_training)
        ax2.set_ylim([0, 0.3])
        ax2.legend(prop={"size": 7})
    elif (discriminative_losses_through_training is not None) and len(
        discriminative_losses_through_training
    ) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
        ax2.set_xlabel("iterations")
        ax2.set_title("Discriminator losses")
        ax2.plot(iters, discriminative_losses_through_training)
        ax2.set_ylim([-0.01, 0.1])
        ax2.legend(prop={"size": 7})
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 7))
    ax1.set_xlabel("iterations")
    ax1.set_title("Marginal 2-Wasserstein errors")
    ax1.plot(iters, marginal_w2_losses_through_training, label=single_coord_labels)
    ax1.set_ylim([0, 0.03])
    ax1.legend(prop={"size": 7})

    # Add the descriptor
    graph_filename = graph_filename + "_{}.png".format(descriptor)
    # graph_filename = f"model_saves/{self.dict_saves_folder}/graph_{self.dict_saves_folder}_num{self.serial_number}_{descriptor}.png"
    fig.savefig(graph_filename)


def sample_fake_data(generator, bm_in):
    """
    Takes in w and prints out levy area.
    :param generator:
    :param bm_in:
    :return:
    """
    actual_bsz = bm_in.shape[0]
    generator.eval()
    # self.start_time = timeit.default_timer()
    fake_data = generator(bm_in).detach()
    generator.train()
    return fake_data


def check_perm_inv_densities(
    gen: Generator, h_in: list[float] = None, only_draw_first_dim=False
):
    if h_in is None:
        h_in = [1.0, -2.0, 3.0, -0.5]
    assert len(h_in) == 4
    bsz = 65536
    bm_dim = 4
    levy_dim = int((bm_dim * (bm_dim - 1)) // 2)
    h_in2 = [h_in[1], h_in[0], h_in[2], h_in[3]]

    h1_row = torch.tensor([h_in], dtype=torch.float, device=gen.device)
    h2_row = torch.tensor([h_in2], dtype=torch.float, device=gen.device)

    h1 = h1_row.expand(bsz, -1)
    h2 = h2_row.expand(bsz, -1)

    bb1 = gen.generate_bb(h_in=h1)[:, bm_dim:]
    bb2 = gen.generate_bb(h_in=h2)[:, bm_dim:]
    bb2_prime = torch.clone(bb2)

    # triu_indices = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    bb2_prime[:, 0] = -bb2[:, 0]
    bb2_prime[:, 1] = bb2[:, 3]
    bb2_prime[:, 3] = bb2[:, 1]
    bb2_prime[:, 2] = bb2[:, 4]
    bb2_prime[:, 4] = bb2[:, 2]

    errors = [
        sqrt(ot.wasserstein_1d(bb1[:, i], bb2_prime[:, i], p=2))
        for i in range(levy_dim)
    ]
    print(make_pretty(errors))
    distns1 = [bb1[:, i].detach().to("cpu").numpy() for i in range(6)]
    distns2 = [bb2_prime[:, i].detach().to("cpu").numpy() for i in range(6)]

    if only_draw_first_dim:
        fig, ax = plt.subplots(
            nrows=1, ncols=1, sharex="all", sharey="all", figsize=(5, 3)
        )
        ax.set_title("b(0,1)")
        labels = ["original", "1st and 2nd dim swapped"]
        ax.hist(
            (distns1[0], distns2[0]),
            bins=400,
            density=True,
            histtype="step",
            stacked=False,
            label=labels,
        )
        ax.set_xlim(-1.5, 1.5)
        ax.legend(prop={"size": 6})
        fig.show()
        return

    fig, axs = plt.subplots(
        nrows=2, ncols=3, sharex="all", sharey="all", figsize=(16, 5)
    )

    def axx(i):
        return axs[i // 3, i % 3]

    triu_indices = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    for i in range(6):
        ax: plt.Axes = axx(i)
        ax.set_title(f"b({triu_indices[i][0] + 1}, {triu_indices[i][1] + 1})")
        labels = ["original", "1st and 2nd dim swapped"]
        ax.hist(
            (distns1[i], distns2[i]),
            bins=400,
            density=True,
            histtype="step",
            stacked=False,
            label=labels,
        )
        ax.set_xlim(-1.5, 1.5)
        ax.legend(prop={"size": 6})

    fig.show()


def draw_density_graphs(
    data_in: np.ndarray, use_levy_dim_titles=False, xlim: float = 1.5
):
    """
    Draws histograms of data_in
    Args:
        data_in (np.ndarray):
        use_levy_dim_titles: Just to ensure proper titling of graphs when graphing levy data

    Returns:

    """
    num_features = data_in.shape[1]
    graph_arrangement = {
        1: (1, 1),
        2: (1, 2),
        3: (1, 3),
        4: (2, 2),
        5: (1, 5),
        6: (2, 3),
        7: (2, 4),
        10: (2, 5),
        15: (3, 5),
        21: (5, 7),
    }

    nrows, ncols = graph_arrangement[num_features]
    print(ncols)
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex="all",
        sharey="all",
        figsize=(5 * ncols, 3 * nrows),
    )

    def axx(i):
        return axs[i // ncols, i % ncols]

    for i in range(num_features):
        ax: plt.Axes = axx(i)
        if use_levy_dim_titles:
            bm_dim = int((1 + sqrt(1 + 8 * num_features)) / 2)
            ax.set_title(f"dim {bm_indices(i, bm_dim)}")
        else:
            ax.set_title(f"dim {i}")
        ax.hist(data_in[:, i], bins=400, density=True, histtype="step", stacked=False)
        ax.set_xlim(-xlim, xlim)

    fig.show()


def draw_generator_densities(
    gen: Generator,
    generate_bb=False,
    data_in: torch.Tensor = None,
    data_in_row: list[float] = None,
    bsz: int = 65536,
    xlim: float = 1.5,
):
    """
    Draws density graphs of the generator's outputs.
    Args:
        bsz:
        gen: the generator
        generate_bb: If False will generate Levy area given W=data_in, if True will generate bb given H=data_in
        data_in: either W increment or H increment (as a tensor)
        data_in_row:
    """
    if data_in is not None:
        bsz = data_in.shape[0]
    bm_dim = gen.bm_dim
    data = data_in
    if data_in_row is not None and data is None:
        data: torch.Tensor = torch.tensor(
            [data_in_row], dtype=torch.float, device=gen.device
        )
        data = data.expand(bsz, -1)

    if data is None:
        data = torch.randn((bsz, bm_dim), dtype=torch.float, device=gen.device)

    assert bm_dim == data.shape[1]

    if generate_bb:
        out = gen.generate_bb(h_in=data)[:, bm_dim:]
    else:
        out = gen.forward(data)[:, bm_dim:]
    out_numpy = out.detach().to("cpu").numpy()
    draw_density_graphs(out_numpy, use_levy_dim_titles=True, xlim=xlim)


def compute_objectives(
    levG, gen_conf: dict, tr_conf: dict, num_trials: int = 5, use_saves=False
):
    files = [""] * num_trials
    if use_saves:
        levG.init_generator(gen_conf)
        levG.generator.init_dict_saves_folder()
        path = f"model_saves/{levG.generator.dict_saves_folder}/"
        files = [
            path + f
            for f in listdir(path)
            if isfile(join(path, f)) and ("best__scr" in f or "best_4mom" in f)
        ]
        print(len(files))

    min_marginal_loss = 100.0
    min_eval_time = 100.0
    min_mom4_loss = 100.0
    avg_marginal_loss = 0
    avg_eval_time = 0
    avg_mom4_loss = 0

    for trial in range(len(files)):
        levG.reset_test_results()
        if use_saves:
            levG.generator.load_dict(filename=files[trial])
        else:
            levG.init_generator(gen_conf)

        levG.cf_train(tr_conf)
        results = levG.tester.test_results

        err = 0.01 * results["best score"]

        mom4_loss = results["best_max_4mom_errors"]

        z = torch.randn((65536, levG.bm_dim), dtype=torch.float, device=levG.device)
        start_time = default_timer()
        levG.generator.sample_fake_data(z)
        elapsed = default_timer() - start_time

        min_marginal_loss = err if err < min_marginal_loss else min_marginal_loss
        avg_marginal_loss += err / num_trials

        min_mom4_loss = mom4_loss if mom4_loss < min_mom4_loss else min_mom4_loss
        avg_mom4_loss += mom4_loss / num_trials

        min_eval_time = elapsed if elapsed < min_eval_time else min_eval_time
        avg_eval_time += elapsed / num_trials

        print(
            f"trial: {trial}, elapsed: {make_pretty(elapsed)}, marginal_err: {make_pretty(err)},"
            f" 4mom_err: {make_pretty(mom4_loss)}"
        )

    marginal_err = avg_marginal_loss - min_marginal_loss
    mom4_err = avg_mom4_loss - min_mom4_loss
    time_err = avg_eval_time - min_eval_time

    return (
        min_marginal_loss,
        marginal_err,
        min_mom4_loss,
        mom4_err,
        min_eval_time,
        time_err,
    )


def full_evaluation(
    x_real,
    generator,
    conf,
    device,
    foster=False,
    gamma=0.5,
    num_iter=5,
    real_fourth_moments=None,
):
    """
    Generate unconditional Levy process, compute the test metrics and
    :param x_real: real levy area and increment generated by Julia package
    :param generator: trained LevyGAN
    :param conf: generator configuration file
    :param device: cuda or cpu
    :return: loss_dict: dictionary with test metrics
    """

    bm_dim = conf["bm_dim"]
    levy_dim = int(bm_dim * (bm_dim - 1) / 2)
    bm_real = x_real[:, :bm_dim]
    levy_real = x_real[:, bm_dim:]
    if not real_fourth_moments:
        real_fourth_moments = fourth_moments(levy_real)
    # Initialize the kernel
    kernel = RBFKernel(gamma=gamma)
    mmd = MMD(kernel, sub_matrix_dim=2**10)

    loss_dict = {
        "marginal_w2_losses": [],
        "fourth_moment_losses": [],
        "MMD": [],
        "unconditional_loc": [],
        "unconditional_scale": [],
    }

    with torch.no_grad():
        # Repeat the generation for 5 times
        for seed in range(num_iter):
            torch.manual_seed(seed)
            mmd = MMD(kernel, sub_matrix_dim=2**10)
            if not foster:
                x_fake = generator(bm_real)
            else:
                x_fake = mom4_gpu(bm_dim, x_real.shape[0], bm_in=bm_real)
            levy_fake = x_fake[:, bm_dim:]
            marginal_w2_loss = np.mean(
                np.array(
                    [
                        100
                        * sqrt(ot.wasserstein_1d(levy_real[:, i], levy_fake[:, i], p=2))
                        for i in range(levy_dim)
                    ]
                )
            )
            fourth_moment_loss = fourth_moment_errors(
                levy_fake, real_fourth_moments
            ).item()
            loss_dict["marginal_w2_losses"].append(marginal_w2_loss)
            loss_dict["fourth_moment_losses"].append(fourth_moment_loss)
            mmd_value = mmd(x_real, x_fake)
            loss_dict["MMD"].append(mmd_value)
            del x_fake, mmd, levy_fake

        # if not foster:
        #     _, levy_bb = generator.generate_bb(10 ** 5, concat_hb = False)
        #
        #     def fit_logistic(data):
        #         loc, scale = logistic.fit(data)
        #         return loc, scale
        #
        #     fitted_loc, fitted_scale = fit_logistic(levy_bb[:, 0].detach().cpu())
        #     loss_dict['unconditional_loc'].append(fitted_loc)
        #     loss_dict['unconditional_scale'].append(fitted_scale)

    return loss_dict


"""
MMD computation through Kernel
"""


class Kernel:
    """
    Base class
    """

    def __call__(self, x, y):
        raise NotImplementedError


class LinearKernel(Kernel):
    """
    Linear Kernel
    """

    def __call__(self, x, y):
        return torch.matmul(x, y.t())


class PolynomialKernel(Kernel):
    """
    Polynomial Kernel
    """

    def __init__(self, degree=3, c=1):
        self.degree = degree
        self.c = c

    def __call__(self, x, y):
        return (torch.matmul(x, y.t()) + self.c) ** self.degree


class RBFKernel(Kernel):
    """
    Radial basis function kernel
    """

    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def __call__(self, x, y):
        xx = torch.sum(x**2, dim=1).view(-1, 1)
        yy = torch.sum(y**2, dim=1).view(1, -1)
        distance = xx + yy - 2.0 * torch.matmul(x, y.t())
        return torch.exp(-self.gamma * distance)


class MMD:
    def __init__(self, kernel, sub_matrix_dim=None):
        self.kernel = kernel
        self.sub_matrix_dim = sub_matrix_dim

    def mmd(self, X, Y):
        """
        Computes E(K(X,Y)), batchwise if needed
        :param X:
        :param Y:
        :return:
        """
        m = X.size(0)
        n = Y.size(0)
        if self.sub_matrix_dim is None:
            XY = self.kernel(X, Y).sum().item() / (m * n)
        else:
            row_iters = m // self.sub_matrix_dim + 1
            col_iters = n // self.sub_matrix_dim + 1
            XY = 0.0
            for row_num in range(row_iters):
                for col_num in range(col_iters):
                    X1 = X[
                        row_num * self.sub_matrix_dim : (row_num + 1)
                        * self.sub_matrix_dim
                    ]
                    Y1 = Y[
                        col_num * self.sub_matrix_dim : (col_num + 1)
                        * self.sub_matrix_dim
                    ]
                    XY += self.kernel(X1, Y1).sum().item()

            XY /= m * n
        return XY

    def __call__(self, X, Y):
        return self.mmd(X, X) + self.mmd(Y, Y) - 2 * self.mmd(X, Y)
