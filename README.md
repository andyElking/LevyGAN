# LevyGAN

In this project, we provide the code pipeline for training and evaluation of a deep-learning-based model for generating approximate samples of the Lévy area conditional on a Brownian increment.

## Environment Setup

The code has been tested successfully using Python 3.8 and pytorch 1.11.0. A typical process for installing the package dependencies involves creating a new Python virtual environment.

To install the required packages, run the following:

```console
pip install -r requirements.txt
```

In order to use the provided testing functionality, please download the Levy area sample datasets from the link below and copy them into LevyGAN/samples/.

https://drive.google.com/drive/folders/1GZ0MY_9BMeg6Tkcu5BrHeENDBn1TGPjM?usp=sharing


## The LevyGAN Object
The `LevyGAN.py` file contains the base class `LevyGAN`. Only the dimension of Brownian motion to be used for training and testing needs to be provided. The `LevyGAN` object has four main attributes which are initialised separately:

- `LevyGAN.generator`, initialised using `LevyGAN.init_generator(gen_config)`. The dictionary gen_config contains all relevant information pertaining to the generator setup, i.e. dimension of latent noise, hyperparameters of the network etc.
- `LevyGAN.disciminator`, initialised using `LevyGAN.init_discriminator(discr_config)`. The dictionary gen_config contains all relevant information pertaining to the discriminator setup, i.e. whether to use a discriminator based on the true joint characteristic function of Brownian motion and Lévy area, or the unitary characteristic function, the number of points at which to evaluate the characteristic function, the degree of the unitary Lie algebra if using the unitary characteristic discriminator etc.
- `LevyGAN.trainer`, initialised using `LevyGAN.init_trainer(trainer_config)`. The dictionary trainer_config contains all relevant information pertaining to the training setup, i.e. the training batch size, the number of generator updates per discriminator update, as well as type of discriminator used. There are two types of trainer: CF_trainer, which uses the joint characteristic function, and UCF_trainer, which uses the unitary characteristic function. The trainer object provides the functionality of a one step-update of the generator and discriminator in the method `LevyGAN.trainer.step_fit()`.
- `LevyGAN.tester`, initialised using `LevyGAN.init_tester(tester_config)`. The dictionary tester_config contains all relevant information pertaining to the testing setup, i.e. the number of test samples to use, whether training graphs should be generated at end of training etc. The main functionality of the tester is to keep track of several key evaluation metrics such as the marginal Wasserstein 2-error, and a joint fourth moment error during training. These metrics are used to select the optimal model.

The `LevyGAN.fit()` method takes as arguments, the generator, discriminator, trainer and tester and performs the training procedure. It calls the trainer once per epoch to perform an update of the generator and discriminator, and, in-line with  the specified testing frequency, calls the tester to evaluate the current model on a set of benchmark metrics.


## Model training

We provide a general interface `run.py` for model training and evaluation. 

The function `model_training()` in `run.py` performs the model training. The function takes 5 inputs, corresponding to the configuration dictionaries of the GAN, generator, discriminator, trainer, and tester. Sample configurations can be found in `/configs_folder`. During the training, the folders `/model_saves` and `/graphs` will be created if the user decides to store the trained model and training graphs.

The trained model will be saved under the  directory `/model_saves` and the graphs created during the training will be saved in `/graphs`. The model's name will be given by `descriptor`, a member variable of ``

## Model evaluation

Once the model is trained, use the function `model_evalution()` in `run.py` to assess the model performance. The function takes as the input the directory where the model is saved and the generator configuration. The returning object is a dictionary that contains the different test metrics listed in the article. In order to use this functionality, Levy area samples have to be downloaded from the drive link above. Alternatively, the user may generate their own sample datasets using the Julia-language script in classical_sample_generator.jl.

## Multilevel Monte Carlo

We conducted a numerical analysis on the log-Heston model, a popular SDE in mathematical finance, demonstrating that high-quality synthetic Lévy area can lead to high-order weak
convergence and variance reduction when using multilevel Monte Carlo (MLMC).

We implement the Strang log-ODE scheme suggested by [Foster, dos Reis and Strang](https://arxiv.org/abs/2210.17543) which involves a fake Lévy area term. We first compare this to the classical Milstein scheme as well as the Milstein antithetic scheme of [Giles and Szpruch](https://projecteuclid.org/journals/annals-of-applied-probability/volume-24/issue-4/Antithetic-multilevel-Monte-Carlo-estimation-for-multi-dimensional-SDEs-without/10.1214/13-AAP957.full). This comparison demonstrates that the Strang log-ODE scheme achieves a higher order variance reduction and weak convergence rate than two popular schemes which do not use Lévy area.

We then compare the impact of different estimators for the fake Lévy area: our generative method, [Foster's](https://ora.ox.ac.uk/objects/uuid:775fc3f5-501c-425f-8b43-fc5a7b2e4310) moment matching method, and a Rademacher random variable independent of the Brownian increment that matches the variance of Lévy area. We repeat the experiments a number of times and shade the standard deviation of the log-error in the plots.
