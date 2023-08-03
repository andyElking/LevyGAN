import numpy as np
import torch
from torch import nn
import math
from functools import partial

def matrix_power_two_batch(A, k):
    orig_size = A.size()
    A, k = A.flatten(0, -3), k.flatten()
    ksorted, idx = torch.sort(k)
    # Abusing bincount...
    count = torch.bincount(ksorted)
    nonzero = torch.nonzero(count, as_tuple=False)
    A = torch.matrix_power(A, 2 ** ksorted[0])
    last = ksorted[0]
    processed = count[nonzero[0]]
    for exp in nonzero[1:]:
        new, last = exp - last, exp
        A[idx[processed:]] = torch.matrix_power(
            A[idx[processed:]], 2 ** new.item())
        processed += count[exp]
    return A.reshape(orig_size)


def rescaled_matrix_exp(f, A):
    """
    An efficient way of computing the tensor exponential
    :param f:
    :param A:
    :return:
    """
    normA = torch.max(torch.sum(torch.abs(A), axis=-2), axis=-1).values
    more = normA > 1
    s = torch.ceil(torch.log2(normA)).long()
    s = normA.new_zeros(normA.size(), dtype=torch.long)
    s[more] = torch.ceil(torch.log2(normA[more])).long()
    A_1 = torch.pow(
        0.5, s.float()).unsqueeze_(-1).unsqueeze_(-1).expand_as(A) * A
    # print(A_1.shape)
    return matrix_power_two_batch(f(A_1), s)


def unitary_lie_init_(tensor: torch.tensor, init_=None):
    r"""Fills in the input ``tensor`` in place with initialization on the unitary Lie.
    Since the
    The blocks are of the form
    :math:`\begin{pmatrix} 0 & b \\ -b & 0\end{pmatrix}` where :math:`b` is
    distributed according to ``init_``.
    This matrix is then projected onto the manifold using ``triv``.
    The input tensor must have at least 2 dimension. For tensors with more than 2 dimensions
    the first dimensions are treated as batch dimensions.
    Args:
        tensor (torch.Tensor): a 2-dimensional tensor
        init\_ (callable): Optional. A function that takes a tensor and fills
                it in place according to some distribution. See
                `torch.init <https://pytorch.org/docs/stable/nn.init.html>`_.
                e.g. partial(nn.torch.uniform_,a=-1,b=1)
                Default: :math:`\operatorname{Uniform}(-1, 1)`
    """
    if tensor.ndim < 2 or tensor.size(-1) != tensor.size(-2):
        raise ValueError(
            "Only tensors with 2 or more dimensions which are square in "
            "the last two dimensions are supported. "
            "Got a tensor of shape {}".format(tuple(tensor.size()))
        )

    n = tensor.size(-2)
    tensorial_size = tensor.size()[:-2]

    # Non-zero elements that we are going to set on the diagonal
    n_diag = n
    diag = tensor.new(tensorial_size + (n_diag,))
    if init_ is None:
        torch.nn.init.uniform_(diag, -math.pi, math.pi)
    else:
        init_(diag)
    diag = diag.imag*torch.tensor([1j], device=tensor.device)
    # set values for upper trianguler matrix
    off_diag = tensor.new(tensorial_size+(2*n, n))
    if init_ is None:
        torch.nn.init.uniform_(off_diag, -math.pi, math.pi)

    else:
        init_(off_diag)

    upper_tri_real = torch.triu(off_diag[..., :n, :n], 1).real.cfloat()
    upper_tri_complex = torch.triu(
        off_diag[..., n:, :n], 1).imag.cfloat()*torch.tensor([1j], device=tensor.device)

    real_part = (upper_tri_real - upper_tri_real.transpose(-2, -1)
                 )/torch.tensor([2], device=tensor.device).cfloat().sqrt()
    complex_part = (upper_tri_complex + upper_tri_complex.transpose(-2, -1)
                    )/torch.tensor([2], device=tensor.device).cfloat().sqrt()

    with torch.no_grad():
        # First non-central diagonal
        x = real_part+complex_part+torch.diag_embed(diag)
        if unitary(n).in_lie_algebra(x):
            tensor = tensor.cfloat()
            tensor.copy_(x)
            return tensor
        else:
            raise ValueError(
                "initialize not in Lie")


class unitary(nn.Module):
    def __init__(self, size):
        """
        real symplectic lie algebra matrices, parametrized in terms of
        by a general linear matrix with shape (2n,2n ).
        Args:
            size (torch.size): Size of the tensor to be parametrized
        """
        super().__init__()
        self.size = size

    @ staticmethod
    def frame(X: torch.tensor) -> torch.tensor:
        """ parametrise real symplectic lie algebra from the gneal linear matrix X

        Args:
            X (torch.tensor): (...,2n,2n)
            J (torch.tensor): (2n,2n), symplectic operator [[0,I],[-I,0]]

        Returns:
            torch.tensor: (...,2n,2n)
        """
        X = (X - torch.conj(X.transpose(-2, -1)))/2

        return X

    def forward(self, X: torch.tensor) -> torch.tensor:
        if len(X.size()) < 2:
            raise ValueError('weights has dimension < 2')
        if X.size(-2) != X.size(-1):
            raise ValueError('not squared matrix')
        return self.frame(X)

    @ staticmethod
    def in_lie_algebra(X, eps=1e-5):
        return (X.dim() >= 2
                and X.size(-2) == X.size(-1)
                and torch.allclose(torch.conj(X.transpose(-2, -1)), -X, atol=eps))

class projection(nn.Module):
    def __init__(self, input_size, hidden_size, channels=1, init_range=1, **kwargs):
        """this class is used to project the path increments to the Lie group path increments, with Lie algbra trainable weights.
        Args:
            input_size (int): input size of the path
            hidden_size (int): size of the hidden Lie algbra matrix
            channels (int, optional): number of channels, produce independent Lie algebra weights. Defaults to 1.
            param (method, optional): parametrization method to map the GL matrix to required matrix Lie algebra. Defaults to sp.
            triv (function, optional): the trivialization map from the Lie algebra to its correpsonding Lie group. Defaults to expm.
        """
        self.__dict__.update(kwargs)

        A = torch.empty(input_size, channels, hidden_size,
                        hidden_size, dtype=torch.cfloat)
        self.channels = channels
        super(projection, self).__init__()
        # self.size = hidden_size
        self.param_map = unitary(hidden_size)
        self.param = unitary
        # A = self.M_initialize(A)
        self.A = nn.Parameter(A)

        self.triv = torch.matrix_exp
        self.init_range = init_range
        self.reset_parameters()

        self.hidden_size = hidden_size

    def reset_parameters(self):
        unitary_lie_init_(self.A, partial(nn.init.normal_, std=1))


    def M_initialize(self, A):
        init_range = np.linspace(0, 10, self.channels+1)
        for i in range(self.channels):
            A[:, i] = unitary_lie_init_(A[:, i], partial(nn.init.uniform_,
                                                         a=init_range[i], b=init_range[i+1]))
        return A

    # @jit.script_method

    def forward(self, dX: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dX (torch.tensor): (N,input_size) the path increment
        Returns:
            torch.tensor: (N,channels,hidden_size,hidden_size)
        """
        # self.reset_parameters()
        # We make sure the map yields elements in the Lie algebra
        A = self.param_map(self.A).permute(1, 2, -1, 0)  # C,m,m,in
        AX = A.matmul(dX.T).permute(-1, 0, 1, 2)  # ->C,m,m,N->N,C,m,m

        # out = torch.matrix_exp(AX)
        return rescaled_matrix_exp(self.triv, AX)



class development_layer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, channels: int = 1, init_range=1):
        super(development_layer, self).__init__()
        self.input_size = input_size
        self.channels = channels
        self.hidden_size = hidden_size
        self.projection = projection(
            input_size, hidden_size, channels, init_range=init_range)
        self.complex = True

    # @jit.script_method
    def forward(self, input_path: torch.Tensor) -> torch.Tensor:
        """forward

        Args:
            input_path (torch.tensor): tensor with shape (N, input_size)

        Returns:
            [type]: [description] (N,channels,hidden_size,hidden_size)
        """
        if self.complex:
            input_path = input_path.cfloat()

        N, C = input_path.shape

        M_dX = self.projection(input_path).reshape(N, self.channels, self.hidden_size, self.hidden_size)

        return M_dX
