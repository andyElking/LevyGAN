import equinox as eqx
from jax import random as jr, numpy as jnp, tree_util as jtu

from jax_src.generator import Layer, MultLayer, AbstractLayer, Net


def make_layer(key, in_features, out_features, use_batch_norm, use_activation) -> Layer:
    wkey, bkey = jr.split(key, 2)
    lim = 1 / jnp.sqrt(in_features)
    weight = jr.uniform(wkey, (out_features, in_features), minval=-lim, maxval=lim)
    bias = jr.uniform(bkey, (out_features,), minval=0, maxval=lim)
    return Layer(weight, bias, use_batch_norm, use_activation)


def make_multlayer(
    key, in_features, out_features, use_batchnorm, use_activation
) -> MultLayer:
    del use_batchnorm
    wkey0, wkey1, wkey2, bkey = jr.split(key, 4)
    lim = 1 / jnp.sqrt(in_features)
    weight0 = jr.uniform(
        wkey0, (out_features // 2, in_features), minval=-lim, maxval=lim
    )
    weight1 = jr.uniform(
        wkey1, (out_features // 2, in_features), minval=-lim, maxval=lim
    )
    weight2 = jr.uniform(
        wkey2, (out_features // 2, in_features), minval=-lim, maxval=lim
    )
    bias = jr.uniform(bkey, (out_features,), minval=0, maxval=lim)
    return MultLayer(weight0, weight1, weight2, bias, use_activation)


def make_layers(
    key,
    in_features,
    hidden_size,
    num_layers,
    use_multlayer,
    dtype,
    use_batch_norm=False,
    use_activation=True,
) -> list[AbstractLayer]:
    keys = jr.split(key, num_layers)
    out_features = 1
    make_first_layer = make_multlayer if use_multlayer else make_layer
    layers = [
        make_first_layer(
            keys[0], in_features, hidden_size, use_batch_norm, use_activation
        )
    ]
    for i in range(1, num_layers - 1):
        layers.append(
            make_layer(
                keys[i], hidden_size, hidden_size, use_batch_norm, use_activation
            )
        )
    layers.append(make_layer(keys[-1], hidden_size, out_features, False, False))

    # Change everything to the right dtype
    layers = jtu.tree_map(lambda x: x.astype(dtype), layers)
    return layers


def make_net(
    key,
    noise_size,
    hidden_size,
    num_layers,
    slope,
    use_multlayer,
    dtype,
    use_batch_norm=False,
    use_activation=True,
) -> Net:
    in_features = 2 * (noise_size + 1)
    layers = make_layers(
        key,
        in_features,
        hidden_size,
        num_layers,
        use_multlayer,
        dtype,
        use_batch_norm,
        use_activation,
    )
    return Net(layers, slope=slope)


def net_name(net):
    nsz = net.noise_size
    num_layers = len(net.layers)
    hidden_dim = net.layers[0].out_features
    mult_layer = "_mult_layer" if isinstance(net.layers[0], MultLayer) else ""
    return f"net_nsz{nsz}_nl{num_layers}_hd{hidden_dim}{mult_layer}"


def save_net(net, path):
    # Name the file using the net_name function
    file_name = path + net_name(net) + ".pckl"
    eqx.tree_serialise_leaves(file_name, net)


def load_net(
    path,
    noise_size,
    hidden_size,
    num_layers,
    slope,
    use_multlayer: bool,
    dtype,
    use_batch_norm=False,
    use_activation=True,
) -> Net:
    multlayer = "_mult_layer" if use_multlayer else ""
    name = f"net_nsz{noise_size}_nl{num_layers}_hd{hidden_size}{multlayer}.pckl"
    file_name = path + name
    mould = make_net(
        jr.PRNGKey(0), noise_size, hidden_size, num_layers, slope, use_multlayer, dtype
    )
    return eqx.tree_deserialise_leaves(file_name, mould)
