import torch

"""
This file includes the functions that return the true joint characteristic function of brownian motion and levy area as well as the empirical characteristic function of a collection of observations.
"""

def sum_A(dim, coeffs=None):
    '''
    The matrix representation of the Levy area sum process A = \sum lambda_i A_i, where each A_i denotes the Levy area matrix of Brownian motion
    '''

    use_coeffs = coeffs is not None
    if use_coeffs:
        assert dim * (dim - 1) / 2 == coeffs.shape[1], "Dimension does not agree"
        batch_size = coeffs.shape[0]
        # coeffs = torch.reshape(coeffs, (batch_size, -1,))
    else:
        batch_size = 1
        coeffs = torch.ones(batch_size, dim * (dim - 1) / 2)
    A = torch.zeros([batch_size, dim, dim]).to(coeffs.device)
    k = 0
    for i in range(dim):
        for j in range(i + 1, dim):
            A[:, i, j] = coeffs[:, k]
            k += 1
            A[:, j, i] = -A[:, i, j]
    return A


def block_diagonal_decomposition(skew_symmetric):
    '''
    Decomposition of any real-valued skew-symmetric matrix A into A = U Sigma U^T where U is an orthogonal matrix and Sigma is block diagonal
    args:
        skew_symmetric: torch.tensor, skew_symmetric is a tensor of the shape [batch, dim, dim]
    '''
    batch_size, dim = skew_symmetric.shape[0], skew_symmetric.shape[1]
    threshold = 1e-5
    assert (torch.norm(skew_symmetric + torch.permute(skew_symmetric, [0, 2, 1])) < threshold), skew_symmetric
    # assert (torch.norm(skew_symmetric) > 1e-10), "A is zero matrix"
    n = dim // 2
    _, evecs = torch.linalg.eig(skew_symmetric)
    evals = torch.linalg.eigvals(skew_symmetric)

    V = evecs.clone()  # Change the eigenvectors correspondently
    # Sort eigenvalues by the norm of the imaginary part
    _, keys = torch.sort(torch.abs(evals.imag), descending=True, axis=1)
    evals = torch.gather(evals, 1, keys)

    gather_keys = keys.reshape(keys.shape[0], 1, -1).repeat(1, keys.shape[1], 1)
    V = torch.gather(V, 2, gather_keys)

    # Make sure that lambda i comes before than -lambda i
    keys = torch.tensor([list(range(evals.shape[1]))]).repeat(evals.shape[0], 1).to(device=skew_symmetric.device)
    keys[:, ::2][(evals[:, ::2].imag < 0)] += 1
    keys[:, 1::2][(evals[:, 1::2].imag > 0)] -= 1
    gather_keys = keys.reshape(keys.shape[0], 1, -1).repeat(1, keys.shape[1], 1).to(device=skew_symmetric.device)
    V = torch.gather(V, 2, gather_keys)

    S = torch.diag_embed(evals)

    W_real = torch.tensor([[0., 1.], [1., 0.]])
    W_img = torch.tensor([[1., 0.], [0., 1.]])
    W = 1 / torch.sqrt(torch.tensor(2)) * torch.complex(W_real, W_img).to(device=skew_symmetric.device,
                                                                          dtype=S.dtype).unsqueeze(0)
    J = - torch.tensor([[0., 1.], [-1., 0.]]).to(device=skew_symmetric.device, dtype=S.dtype).unsqueeze(0).repeat(
        batch_size, 1, 1)

    # Initialize the matrices Omega and Sigma
    Sigma = torch.zeros(batch_size, dim, dim).to(device=skew_symmetric.device, dtype=S.dtype)
    Omega = torch.zeros(batch_size, dim, dim).to(device=skew_symmetric.device, dtype=S.dtype)

    # We need to normalize those eigenvectors with imaginary eigenvalues
    for i in range(n):
        # We need to split into two cases, imaginary eigenvalues and zero eigenvalues
        index = evals[:, 2 * i].imag != 0
        pos_cases = index.sum()
        neg_cases = (~index).sum()

        a_k = torch.norm(evals[:, 2 * i].unsqueeze(0), dim=0).reshape(batch_size, 1, 1)
        Sigma[:, 2 * i:2 * (i + 1), 2 * i:2 * (i + 1)] = a_k * J
        # If the eigenvalue is imaginary, rescale the corresponding eigenvectors by 1/sqrt(-i)
        V[index, :, 2 * i:2 * (i + 1)] *= 1 / torch.sqrt(torch.tensor(-1j))
        # If the eigenvalue is non-zero, we add a diagonal matrix J
        Omega[index, 2 * i:2 * (i + 1), 2 * i:2 * (i + 1)] = W.repeat(pos_cases, 1, 1)
        # Otherwise we add identity matrix
        Omega[~index, 2 * i:2 * (i + 1), 2 * i:2 * (i + 1)] = torch.eye(2).to(device=skew_symmetric.device,
                                                                              dtype=S.dtype).unsqueeze(0).repeat(
            neg_cases, 1, 1)

    # Corner case
    if dim % 2 == 1:
        Omega[:, -1, -1] = torch.ones([batch_size]).to(device=skew_symmetric.device, dtype=S.dtype)
        Sigma[:, -1, -1] = torch.zeros([batch_size]).to(device=skew_symmetric.device, dtype=S.dtype)

    U = V @ Omega.conj().permute([0, 2, 1])
    return U.to(dtype=skew_symmetric.dtype), evals


def get_real_characteristic():
    '''
        Compute the joint characteristic function of d-dimensional Brownian motion and the corresponding Levy terms conditioned on zero starting point.
        bm_coeff has shape [Batch, d]
        levy_coeff has shape [Batch, d*(d-1)/2]
        The output is a long tensor of size Batch, corresponding to the joint characteristic funciton
    '''

    def joint_characteristic_function(coefficients, bm_dim, t=1.0):
        '''
        args:
            coefficients: torch.tensor, input coefficients for joint characteristic function
            bm_dim: int, dimension of BM
            t: float, stopping time of BM
        return:
            res: torch.tensor, joint characteristic function
        '''
        threshold = 1e-5
        batch_size = coefficients.shape[0]
        bm_coeff = coefficients[:, :bm_dim]
        levy_coeff = coefficients[:, bm_dim:]

        # Create and decompose the matrix
        A = sum_A(bm_dim, levy_coeff)
        U, evals = block_diagonal_decomposition(A)
        evals_img = evals.imag[:, ::2]

        loop_time = bm_dim // 2
        cosh_coeff = 1 / torch.cosh(0.5 * t * evals_img)
        z = torch.pow((U @ bm_coeff.to(dtype=U.dtype).unsqueeze(-1)).reshape(batch_size, -1), 2)

        res = torch.ones([batch_size]).to(coefficients.device)
        for i in range(loop_time):
            temp = torch.zeros([batch_size]).to(coefficients.device)
            # Find the indices where the eigenvalue is not zero
            index_set = evals_img[:, i] != 0.

            temp[index_set] = cosh_coeff[index_set, i] * torch.exp(
                - (z[index_set, 2 * i] + z[index_set, 2 * i + 1]) * torch.tanh(0.5 * t * evals_img[index_set, i]) /
                evals_img[index_set, i])

            temp[~index_set] = torch.exp(-0.5 * (z[~index_set, 2 * i] + z[~index_set, 2 * i + 1]) * t)
            res *= temp
        # Add the extra independent characteristic
        if bm_dim % 2:
            # Odd case: we add an extra term
            temp = torch.exp(- 0.5 * (z[:, -1]) * t)
            res *= temp
        return res

    return joint_characteristic_function


def get_fake_characteristic(fake_logsignature, just_real = False, unitary = False):
    # TODO: set the input as a long vector and split it inside the function (degree provided)
    '''
    Compute the joint characteristic function under the empirical measure
    Output: [coeff_batch]
    '''
    bm_batch = fake_logsignature.shape[0]
    if unitary:
        def cal(coefficients, t=1.0):
            '''
            args:
                coefficients: torch.tensor, input coefficients for joint characteristic function
                t: float, stopping time of BM
            return:
                res: torch.tensor, joint characteristic function
            '''
            batch_coefficients = coefficients.repeat(bm_batch, 1, 1).permute([1, 0, 2])  # [coeff_batch, bm_batch, dim]
            coeff_batch = coefficients.shape[0]
            i = torch.tensor(1j)
            res = ((fake_logsignature.repeat([coeff_batch, 1, 1]) * batch_coefficients).sum(-1)).to(dtype=i.dtype)
            res = 1 / bm_batch * torch.exp(i * res).sum(-1)
            return res.real

    if just_real:
        def cal(coefficients, t=1.0):
            '''
            args:
                coefficients: torch.tensor, input coefficients for joint characteristic function
                t: float, stopping time of BM
            return:
                res: torch.tensor, joint characteristic function
            '''
            batch_coefficients = coefficients.repeat(bm_batch, 1, 1).permute([1, 0, 2])  # [coeff_batch, bm_batch, dim]
            coeff_batch = coefficients.shape[0]
            i = torch.tensor(1j)
            res = ((fake_logsignature.repeat([coeff_batch, 1, 1]) * batch_coefficients).sum(-1)).to(dtype=i.dtype)
            res = 1 / bm_batch * torch.exp(i * res).sum(-1)
            return res.real
    else:
        def cal(coefficients, t=1.0):
            '''
            args:
                coefficients: torch.tensor, input coefficients for joint characteristic function
                t: float, stopping time of BM
            return:
                res: torch.tensor, joint characteristic function
            '''
            batch_coefficients = coefficients.repeat(bm_batch, 1, 1).permute([1, 0, 2])  # [coeff_batch, bm_batch, dim]
            coeff_batch = coefficients.shape[0]
            i = torch.tensor(1j)
            res = ((fake_logsignature.repeat([coeff_batch, 1, 1]) * batch_coefficients).sum(-1)).to(dtype=i.dtype)
            res = 1 / bm_batch * torch.exp(i * res).sum(-1)
            return res

    return cal

