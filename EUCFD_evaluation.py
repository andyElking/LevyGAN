import seaborn as sns
import torch

from src.evaluation.evaluation import *
from src.model.Generator import PairNetGenerator
from src.model.discriminator.Discriminator import UCF_Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sns.set()


def optimize_CF(
    X_dl: torch.tensor,
    Y_dl: torch.tensor,
    char_func,
    iterations: int,
    device,
):
    char_func.to(device)
    best_loss = 0.0
    losses = []
    char_optimizer = torch.optim.Adam(char_func.parameters(), betas=(0, 0.9), lr=0.02)
    print("start opitmize charateristics function")
    for i in tqdm(range(iterations)):
        char_func.train()
        char_optimizer.zero_grad()
        X = next(iter(X_dl)).to(device)
        Y = next(iter(Y_dl)).to(device)
        char_loss = -char_func.empirical_char_diff(X, Y)
        if -char_loss > best_loss:
            print("Loss updated: {}".format(-char_loss))
            best_loss = -char_loss

        losses.append(-char_loss.item())
        # print(char_loss)
        # char_loss = - self.char_func.distance_measure(
        #   self.D(x_real), self.D(x_fake))
        char_loss.backward()
        char_optimizer.step()

    trained_char_func = char_func
    return trained_char_func, losses


def eucfd(generator_name, bm_dim):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if generator_name == "net":
        from configs_folder.configs import gen_config

        generator = PairNetGenerator(gen_conf=gen_config)
        params = torch.load(
            "./good_model_saves/generator_4d_PairNet3LAY_16HID_lky0.01lky0.01ACT_BN_4noise_bf/gen_num5_best__scr.pt"
        )

        # Load the model
        for i, layer in enumerate(generator.layer_list):
            layer.load_state_dict(params[i])
        generator.to(device)
        generator.eval()
        generator.do_bridge_flipping = True

    x_real = torch.tensor(
        np.genfromtxt(f"samples/samples_{bm_dim}-dim.csv", dtype=float, delimiter=",")
    ).to(dtype=torch.float, device=device)[:200000]
    print(x_real.shape)
    levy_real = x_real[:, bm_dim:]
    levy_dim = int(bm_dim * (bm_dim - 1) / 2)

    bm_real = x_real[:, :bm_dim]
    levy_real = x_real[:, bm_dim:]
    with torch.no_grad():
        if generator_name == "net":
            x_fake = generator(bm_real)
        elif generator_name == "foster":
            x_fake = mom4_gpu(bm_dim, x_real.shape[0], device_to_use=device)
        else:
            x_fake = Davie_gpu_all(bm_dim, x_real.shape[0])

    levy_fake = x_fake[:, bm_dim:]

    discr_config = {
        "bm_dim": bm_dim,
        "discr_type": "u_characteristic",
        # "grid_characteristic", 'gaussian_characteristic', 'iid_gaussian_characteristic'
        "discr_measure": "Gaussian",  # "Gaussian", 'Cauchy', doesn't matter if u_characteristic selected
        "coeff_batch": 128,  # Number of points at which to evalute characteristic function
        "lie_degree": 8,  # Only matters for u_characteristic
    }

    discriminator = UCF_Discriminator(discr_config).to(device)
    # discriminator.total_dim = levy_real.shape[-1]

    real_dl, fake_dl = (
        DataLoader(x_real[:160000], batch_size=1024, shuffle=True),
        DataLoader(x_fake[:160000], batch_size=1024, shuffle=True),
    )
    # Train UCFD
    trained_char_func, training_loss = optimize_CF(
        real_dl, fake_dl, discriminator, 2000, "cuda"
    )

    torch.save(
        trained_char_func.unitary_representation.state_dict(),
        "./model_saves/ucf_discriminator/discriminator_{}_bm_dim_{}_iter.pt".format(
            generator_name, bm_dim
        ),
    )
    # Test UCFD
    trained_char_func.eval()
    with torch.no_grad():
        training_loss = torch.tensor(training_loss)[-20:].mean().item()
        # training_loss = 0
        test_loss = discriminator.empirical_char_diff(
            x_real[160000:], x_fake[160000:]
        ).item()

    return pd.DataFrame(
        [
            {
                "model": generator_name,
                "train_loss": training_loss,
                "test_loss": test_loss,
                "bm_dim": bm_dim,
            }
        ]
    )


def cross_validation(generator_name, bm_dim):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if generator_name == 'net':
        from configs_folder.configs import gen_config
        generator = PairNetGenerator(gen_conf=gen_config)
        params = torch.load(
            "./good_model_saves/generator_4d_PairNet3LAY_16HID_lky0.01lky0.01ACT_BN_4noise_bf/gen_num5_best__scr.pt")

        # Load the model
        for i, layer in enumerate(generator.layer_list):
            layer.load_state_dict(params[i])
        generator.to(device)
        generator.eval()
        generator.do_bridge_flipping = True

    x_real = torch.tensor(np.genfromtxt(f"samples/samples_{bm_dim}-dim.csv", dtype=float, delimiter=',')).to(
        dtype=torch.float, device=device)[:200000]
    print(x_real.shape)
    levy_real = x_real[:, bm_dim:]
    levy_dim = int(bm_dim * (bm_dim - 1) / 2)

    bm_real = x_real[:, :bm_dim]
    levy_real = x_real[:, bm_dim:]
    with torch.no_grad():
        if generator_name == 'net':
            x_fake = generator(bm_real)
        elif generator_name == 'foster':
            x_fake = mom4_gpu(bm_dim, x_real.shape[0], device_to_use=device)
        else:
            x_fake = Davie_gpu_all(bm_dim, x_real.shape[0])

    levy_fake = x_fake[:, bm_dim:]

    discr_config = {
        'bm_dim': bm_dim,
        'discr_type': "u_characteristic",
        # "grid_characteristic", 'gaussian_characteristic', 'iid_gaussian_characteristic'
        'discr_measure': "Gaussian",  # "Gaussian", 'Cauchy', doesn't matter if u_characteristic selected
        'coeff_batch': 128,  # Number of points at which to evalute characteristic function
        'lie_degree': 8,  # Only matters for u_characteristic
    }

    discriminator = UCF_Discriminator(discr_config).to(device)

    if generator_name == 'net':
        discriminator.unitary_representation.load_state_dict(torch.load("./model_saves/ucf_discriminator/discriminator_foster_bm_dim_{}_iter.pt".format(bm_dim)))
    else:
        discriminator.unitary_representation.load_state_dict(torch.load("./model_saves/ucf_discriminator/discriminator_net_bm_dim_{}_iter.pt".format(bm_dim)))

    discriminator.eval()
    with torch.no_grad():
        # training_loss = 0
        test_loss = discriminator.empirical_char_diff(x_real[160000:], x_fake[160000:]).item()

    return pd.DataFrame([{"model": generator_name, "test_loss": test_loss, "bm_dim": bm_dim}])

if __name__ == "__main__":
    bm_dims = [2, 3, 4, 5, 6, 7, 8]
    generators = ["net", "foster"]
    df_list = []
    for bm_dim in bm_dims:
        for generator in generators:
            for i in range(3):
                torch.manual_seed(i)
                df = cross_validation(generator, bm_dim)
                df_list.append(df)
    df = pd.concat(df_list)
    df.to_csv("./eucfd_cross.csv")
