import torch
import math
import itertools
import functools
import numpy as np
from sympy.utilities.iterables import multiset_permutations

'''
stream_dim = time-length of the path
batch_dim = batch_size
channel_dim = dimension of the path


Every member in the tensor algebra is structured as an array of tensors, corresponding to the i-th level.
Every tensor are two dimensional (batch_dim, channel_dim).
Batch_dim is the same for all levels while channel is exponentially increasing
'''

"""
--------------------- Utility Functions --------------------- 
"""


def total_len_algebra(dim: int, level: int) -> int:
    """
    Calculates the length of a given tensor in truncated tensor algebra
    """
    if dim == 1:
        return level
    return int(dim * (dim ** level - 1) / (dim - 1)) + 1


def delta_coordinate_helper(dim: int, level: int) -> list:
    """
    This function outputs an array of all possible combinations of letters into words of length level
    this helps one to find the corresponding coordinate in tensor of a given word
    """
    if level == 0:
        words = [(),]
        return words
    letters = [i for i in range(dim)]
    words = [i for i in itertools.product(letters, repeat=level)]
    return words


def algebra_coordinate_helper(dim: int, level: int) -> list:
    """
    This function can be seen as the sum of delta_coordinate_helper(dim, i) from i = 1 to i = n
    """
    words = []
    for i in range(0, level + 1):
        words += delta_coordinate_helper(dim, i)
    return words


def algebra_coordinate_calculator(word: tuple, dim: int, restricted=False) -> int:
    """
    This function calculates the coordinate of a given word in the space of tensor algebra truncated upto some level
    """
    # assert type(word) == tuple, word
    index = 0
    for i, n in zip(word, [j for j in range(len(word) - 1, -1, -1)]):
        assert i < dim, "Element not in the tensor"
        index += (i + 1) * (dim ** n) - restricted * (dim ** n)
    if restricted:
        return index
    return index


def algebra_level_calculator(sublevel: int, dim: int, level: int):
    """
    Given the sublevel, calculate the indices corresponding to such level in the vectorized element
    """
    assert sublevel <= level
    idx_start = 0
    idx_end = 1
    for i in range(1,sublevel+1):
        idx_start = idx_end
        idx_end = idx_start + dim**i
    return idx_start, idx_end


def transformer(sig: list) -> torch.Tensor:
    """
    Given sig of the form [level1, ..., leveln], transform it into one batched tensor
    """
    level = len(sig)
    res = sig[0].clone()
    for i in range(1, level):
        res = torch.cat((res, sig[i]), 1)
    return res


def inv_transformer(sig, dim: int, level: int) -> list:
    """
    Given the vectorized signature, transform it to graded structure, each element corresponds the i-th level of the tensor
    """
    length = total_len_algebra(dim, level)
    assert sig.size(-1) == length, "Dimensions do not match"
    res = []
    index = 0
    for i in range(0,level):
        index += dim**i
        res.append(index)
    return list(torch.tensor_split(sig, res, 1))


def structure_checker(sig, dim: int, level: int):
    total_len = total_len_algebra(dim, level)
    if type(sig) == torch.Tensor:
        assert sig.shape[-1] == total_len, "Dimension does not agree"
        sig = inv_transformer(sig, dim, level)
    elif type(sig) == list:
        assert len(sig) == level + 1, "Dimension does not agree"
    else:
        raise Exception("Unknown format")
    return sig


def mult_inner(tensor_at_level: torch.Tensor, arg1: list, arg2: list, level_index: int):
    for j in range(level_index, -1, -1):
        k = level_index - j
        out_view = tensor_at_level.view(arg1[j].size(-2),
                                        arg1[j].size(-1),
                                        arg2[k].size(-1))
        # print(out_view.shape)
        out_view.addcmul_(arg2[k].unsqueeze(-2),
                          arg1[j].unsqueeze(-1))
    return


def mult(sig1: list, sig2: list) -> list:
    """
    Concadenation profuct of two tensors
    We distinguish two cases, where the constant term equals to 1 (signatures) and equals to 2 (logsignature)
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    level = len(sig1)
    res = [torch.zeros(sig1[i].size(), device=device) for i in range(len(sig1))]
    for level_index in range(level - 1, -1, -1):
        tensor_at_level = res[level_index]
        mult_inner(tensor_at_level, sig1, sig2, level_index)
    return res


def addition(sig1: list, sig2: list):
    """
    Addition in the tensor algebra
    """
    assert len(sig1) == len(sig2), "dimension does not match"
    level = len(sig1)
    for level_index in range(level):
        sig1[level_index] += sig2[level_index]
    return


def substraction(sig1: list, sig2: list):
    """
    Substraction in the tensor algebra
    """
    assert len(sig1) == len(sig2), "dimension does not match"
    level = len(sig1)
    for level_index in range(level):
        sig1[level_index] -= sig2[level_index]
    return


def tensor_exp(logsig, dim: int, level: int, output_format = "graded"):
    """
    This function calculates the Tensor exponential of a log signature
    graded: if the input has the graded structure
    """
    assert output_format in ["graded", "tensor"], "Unknown parameter, only graded/tensor are supported."
    logsig = structure_checker(logsig, dim, level)
    assert torch.all(logsig[0][:,0] == 0), "Input is not a logsignature"
    input_channel_size = len(logsig) - 1
    temp = [sublevel.clone() for sublevel in logsig]
    res = [sublevel.clone() for sublevel in logsig]
    res[0][:,0] = 1
    for i in range(2, input_channel_size + 1):
        temp = mult(logsig, temp)
        coeff = 1 / math.factorial(i)
        ith_power = [coeff * ilevel for ilevel in temp]
        addition(res, ith_power)

    if output_format == "graded":
        return res
    else:
        return transformer(res)


def tensor_log(sig, dim: int, level: int, output_format = "graded"):
    """
    This function calculates the Tensor exponential of a log signature
    graded: if the input has the graded structure
    """
    assert output_format in ["graded", "tensor"], "Unknown parameter, only graded/tensor are supported."
    sig = sig.clone()
    sig = structure_checker(sig, dim, level)
    assert torch.all(sig[0][:,0] == 1), "Input is not signature"
    sig[0][:,0] = 0 # Convert it to 1 + t
    # input_channel_size = sig[1].size(-1)
    input_channel_size = len(sig) - 1
    temp = [sublevel.clone() for sublevel in sig]
    res = [sublevel.clone() for sublevel in sig]
    for i in range(2, input_channel_size + 1):
        temp = mult(sig, temp)
        coeff = (-1)**(i-1) / i
        ith_power = [coeff * ilevel for ilevel in temp]
        addition(res, ith_power)

    if output_format == "graded":
        return res
    else:
        return transformer(res)


def lie_product(sig1: list, sig2: list) -> list:
    """
    [X, Y] = XY - YX
    """
    assert torch.all(sig2[0][:,0] == 0), "Input is not logsignature"
    assert torch.all(sig1[0][:,0] == 0), "Input is not logsignature"
    XY = mult(sig1, sig2)
    YX = mult(sig2, sig1)
    res = [XY[i] - YX[i] for i in range(len(sig1))]
    return res


def adjoint(X: list, dim: int, level: int):
    """
    Given the reference point X, calculate ad_X(Y) = [X, Y]
    """
    X = structure_checker(X, dim, level)

    def adjoint_X(Y: list) -> list:
        Y = structure_checker(Y, dim, level)
        nonlocal X
        return lie_product(X, Y)

    return adjoint_X


def ad_power_series(base: list, tangent: list, dim: int, level: int) -> list:
    """
    Given the base point X and the tangent vector Y, calculate (1-e^{-ad_X})/ad_X (Y) truncated up to certain level.
    Both base and tangent are vectorized in the tensor algebra space
    """
    ad_X = adjoint(base, dim, level)
    res = [sublevel.clone() for sublevel in tangent]
    temp = [sublevel.clone() for sublevel in tangent]
    for i in range(1, level + 1):
        temp = ad_X(temp)
        coeff = (-1) ** i / math.factorial(i + 1)
        addition(res, [coeff * ilevel for ilevel in temp])
    return res


def d_exp(base, dim: int, level: int):
    """
    dexp_X = e^X otimes the fraction
    base is represented with tensor algebra basis, in the form of [level_1,...,level_n]
    """
    expx = tensor_exp(base, dim, level)

    def d_exp_X(Y):
        nonlocal expx, base, level
        res = ad_power_series(base, Y, level)
        return mult(expx, res)

    return d_exp_X


def triangular_matrix_addition(matrix: torch.Tensor, submatrix: torch.Tensor, dim: int, level: int, row_level: int, col_level: int):
    """
    Plug the submatrix into the original matrix to construct the upper triangular matrix
    """
    row_start, row_end = algebra_level_calculator(row_level, dim, level)
    col_start, col_end = algebra_level_calculator(col_level, dim, level)
    assert submatrix.shape[1] == row_end-row_start and submatrix.shape[2] == col_end-col_start, "Dimensions do not agree."
    matrix[:,row_start: row_end, col_start: col_end] += submatrix
    return


def ad_X_transposed(base, dim: int, level: int) -> torch.Tensor:
    """
    The matrix representation of ad_X transposed
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base = structure_checker(base, dim, level)
    total_len = total_len_algebra(dim, level)
    res = torch.zeros(total_len, total_len).unsqueeze(0).repeat(base[0].shape[0],1,1).to(device)
    for col_level in range(level, -1, -1):
        for row_level in range(col_level-1, -1, -1):
            # One has to instantiate a new tensor in order to do kron, there are memory issues with pytorch if one does not instantiate
            temp_base = torch.empty((base[col_level - row_level].shape[0], 1, base[col_level - row_level].shape[1])).to(device)
            temp_base[:, 0, :] = base[col_level - row_level]
            submatrix = torch.kron(temp_base, torch.eye(dim ** row_level).unsqueeze(0).to(device)) \
                        - torch.kron(torch.eye(dim ** row_level).unsqueeze(0).to(device), temp_base)
            triangular_matrix_addition(res, submatrix, dim, level, row_level, col_level)
    return res


def d_exp_transposed(base, dim: int, level: int) -> torch.Tensor:
    """
    The matrix representation of the ad_X power series transposed, this yields an upper triangular matrix
    The input is exp(base) \otimes the tensor
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base = structure_checker(base, dim, level)
    total_len = total_len_algebra(dim, level)
    res = torch.eye(total_len).unsqueeze(0).repeat(base[0].shape[0],1,1).to(device)
    ad_X_T = ad_X_transposed(base, dim, level)
    temp = torch.eye(total_len).unsqueeze(0).repeat(base[0].shape[0],1,1).to(device)
    for ilevel in range(1, level + 1):
        temp = torch.matmul(temp, ad_X_T)
        coeff = ((-1)**ilevel)/(math.factorial(ilevel+1))
        res += coeff * temp
    return res


def left_mult_transposed(base, dim: int, level: int) -> torch.Tensor:
    """
    The matrix representation of ad_X transposed
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base = structure_checker(base, dim, level)
    total_len = total_len_algebra(dim, level)
    res = torch.zeros(total_len, total_len).unsqueeze(0).repeat(base[0].shape[0],1,1).to(device)
    for col_level in range(level, -1, -1):
        for row_level in range(col_level, -1, -1):
            # One has to instantiate a new tensor in order to do kron, there are memory issues with pytorch if one does not instantiate
            temp_base = torch.empty((base[col_level - row_level].shape[0], 1, base[col_level - row_level].shape[1])).to(device)
            temp_base[:, 0, :] = base[col_level - row_level]
            submatrix = torch.kron(temp_base, torch.eye(dim ** row_level).to(device))
            triangular_matrix_addition(res, submatrix, dim, level, row_level, col_level)
    return res

"""
--------------------- Another way of implementing pi1 --------------------- 
"""


def shuffle_prod_word(word1: tuple, word2: tuple):
    """
    Shuffle product of two words
    Word1, word2: tuple/arrays of letters
    """
    len1 = len(word1)
    len2 = len(word2)
    out_ = np.zeros(len1 + len2, dtype=object)
    index = (0,) * len1 + (1,) * len2
    res = []
    for mask in multiset_permutations(index):
        mask = np.array(mask)
        np.place(out_, 1 - mask, word1)
        np.place(out_, mask, word2)
        res.append(out_.copy())
    return np.array(res, dtype='int16')


def shuffle_prod_words(word1: tuple, words: tuple):
    """
    Shuffle product of a word to a list of words
    """
    res = []
    for word2 in words:
        res.append(shuffle_prod_word(word1, word2))
    res = np.array(res, dtype='int16')
    res = res.reshape(-1, res.shape[2])
    return np.array(res, dtype='int16')


def shuffle_prod_n_words(words: tuple):
    """
    Shuffle product of all the words in the inpu
    """
    if len(words) == 2:
        return shuffle_prod_word(words[0], words[1])
    shuffled = shuffle_prod_word(words[-2], words[-1])
    for i in range(len(words) - 3, -1, -1):
        shuffled = shuffle_prod_words(words[i], shuffled)
    return shuffled


def word_split_permutation(word: tuple, n: int):
    """
    Given a word, break into n-pieces, record all of scenarios
    """
    res = []
    length = len(word)
    splitted = []

    def local_recur(breaks, n):
        if n == 1:
            breaks_start = [0, ] + breaks
            breaks_end = breaks + [length, ]
            temp = []
            for i, j in zip(breaks_start, breaks_end):
                temp.append(word[i:j])
            res.append(temp)
            return
        else:
            if not breaks:
                for i in range(1, length):
                    breaks.append(i)
                    local_recur(breaks, n - 1)
                    breaks.pop()
            else:
                for i in range(breaks[-1] + 1, length):
                    breaks.append(i)
                    local_recur(breaks, n - 1)
                    breaks.pop()

    local_recur(splitted, n)
    return np.array(res, dtype=object)


def pi1_of_word_permutation(word: tuple, dim: int, level: int):
    """
    This function computes all shuffle products of subwords hence the coefficient of pi1
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    length = len(word)
    res = torch.zeros(total_len_algebra(dim, level), device=device)
    if not length:
        return res
    res[algebra_coordinate_calculator(word, dim)] = 1
    all_permutations = set([i for i in itertools.permutations(word)])
    for single_permutation in all_permutations:
        index = algebra_coordinate_calculator(single_permutation, dim)
        for i in range(2, length + 1):
            splitted = word_split_permutation(single_permutation, i)
            for sub_splitted in splitted:
                shuffled_product = shuffle_prod_n_words(sub_splitted)
                # Count all the elements corresponding to the original word, this is the scalar product of two polynomials
                counts = np.count_nonzero(np.all(shuffled_product == word, axis=1))
                counts *= ((-1) ** (i - 1) / i)
                res[index] += counts
    return res


def pi1_permutation(sig, dim: int, level: int):
    """
    Pi1 function defined in equation 3.2.3 in Reutenauer's book
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if len(sig.shape) == 1:
        sig = sig.unsqueeze(dim=0)
    sig_len = total_len_algebra(dim, level)
    assert sig.shape[1] == sig_len, sig
    words = algebra_coordinate_helper(dim, level)
    res = torch.zeros(sig.shape, device=device)
    for ibatch in range(sig.shape[0]):
        batch_res = torch.zeros(sig_len, device=device)
        for index, value in zip([i for i in range((len(sig[ibatch])))], sig[ibatch]):
            word = words[index]
            fn = pi1_of_word_permutation(word, dim, level)
            batch_res += value * fn
        res[ibatch] += batch_res
    return res


def pi1_matrix(dim: int, level: int) -> list:
    """
    Compute the transformation matrix of the projection map
    Return an array of matrices corresponding to each level
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    res = []
    idx_start = 0
    words = algebra_coordinate_helper(dim, level)
    for ilevel in range(0, level+1):
        level_len = dim**ilevel
        temp_res = torch.zeros(level_len, level_len, device=device)
        level_words = words[idx_start:level_len+idx_start]
        for index, word in zip([i for i in range((len(level_words)))], level_words):
            pi1_word = pi1_of_word_permutation(word, dim, level)
            temp_res[index] = pi1_word[idx_start:level_len+idx_start].clone()
        idx_start += level_len
        res.append(temp_res.t())
    return res


"""
--------------------- Lyndon word staff --------------------- 
"""


def foliage_iter(x):
    """
    Foliage of a binary tree
    """
    if type(x) is int:
        yield x
        return
    assert type(x) is tuple, x
    for i in x:
        for j in foliage_iter(i):
            yield j


def less_expression_lyndon(a, b):
    return tuple(foliage_iter(a)) < tuple(foliage_iter(b))


class KeyFromLess:
    """Adapter for using a `less' function as a key in sort or sorted"""

    def __init__(self, less):
        self.less = less

    @functools.total_ordering
    class Key:
        def __init__(self, less, tree):
            self.tree = tree
            self.less = less

        def __eq__(self, o):
            # often identity will be enough here
            return o.tree == self.tree

        def __lt__(self, o):
            return self.less(self.tree, o.tree)

    def __call__(self, tree):
        return self.Key(self.less, tree)


def lyndon_basis(dim: int, level: int):
    """
    Jeremy's code. Given dimension and the truncated level, find all the Lyndon words with brackets.
    """
    out = [[(i,) for i in range(dim)]]
    n = dim
    for ilevel in range(2, level + 1):
        out.append([])
        for firstLev in range(1, ilevel):
            for x in out[firstLev - 1]:
                for y in out[ilevel - firstLev - 1]:
                    if less_expression_lyndon(x, y) and (firstLev == 1 or not less_expression_lyndon(x[1], y)):
                        out[-1].append((x, y))
                        n += 1
        out[-1].sort(key=KeyFromLess(less_expression_lyndon))
    return out, n


def lyndon_find_bracket(word, basis):
    """
    Given a Lyndon word, find the bracket expression
    basis: the list of all lyndon words
    """
    length = len(word)
    assert length <= len(basis), length
    for lyndon in basis[length - 1]:
        temp = tuple([i for i in foliage_iter(lyndon)])
        if temp == word:
            return lyndon
    assert 0, "it is not a Lyndon word"


def lyndon_product(tree, dim: int, level: int):
    """
    Given the Lyndon word expressed as a tree, multiply out the brackets
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if len(tree) == 1:
        total_len = total_len_algebra(dim, level)
        res = torch.zeros((1, total_len), device=device)
        index = algebra_coordinate_calculator(tree, dim)
        res[0][index] = 1
        res = inv_transformer(res, dim, level)
        return res
    if len(tree) == 2:
        res1 = lyndon_product(tree[0], dim, level)
        res2 = lyndon_product(tree[1], dim, level)
        return lie_product(res1, res2)


def lyndon_matrix(dim: int, level: int):
    """
    Find the matrix that projects Lyndon words to the canonical basis
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    basis, lyndon_len = lyndon_basis(dim, level)
    index = 0
    sig_len = total_len_algebra(dim, level)
    res = torch.zeros(lyndon_len, sig_len, device=device)
    for ilevel in basis:
        for word in ilevel:
            lyndon_prod = transformer(lyndon_product(word, dim, level))
            res[index] = lyndon_prod
            index += 1
    return res.t()


def lyndon_to_algebra(lyndon, dim: int, level: int):
    """
    Transform from Lyndon basis to canonical basis
    """
    if len(lyndon.shape) == 1:
        lyndon = lyndon.unsqueeze(dim=0)
    _, lyndon_len = lyndon_basis(dim, level)
    assert lyndon.shape[1] == lyndon_len, lyndon.shape[1]
    matrix = lyndon_matrix(dim, level)
    return torch.matmul(matrix, lyndon.t()).t()


def algebra_to_lyndon(logsig, dim, level):
    """
    Transform from canonical basis to Lyndon basis by solving the least square problem
    """
    if len(logsig.shape) == 1:
        logsig = logsig.unsqueeze(dim=0)
    sig_len = total_len_algebra(dim, level)
    assert logsig.shape[1] == sig_len, logsig
    matrix = lyndon_matrix(dim, level)
    solution = torch.linalg.lstsq(matrix, logsig.t())
    # print(resid, torch.norm(resid))
    assert torch.norm(solution.residuals.t()) < 1e-2, "It is not a logsignature"
    return solution.solution.t()


def extract_nested_tuples(nested_tuple):
    result = []

    def _recursive_extract(t):
        for item in t:
            if isinstance(item, tuple) and len(item) == 2 and all(isinstance(x, int) for x in item):
                result.append(item)
            elif isinstance(item, tuple):
                _recursive_extract(item)

    _recursive_extract(nested_tuple)
    return tuple(result)


def levy_expand(word):
    """
    Express levy area as linear combination of elements in the signature of BM
    Transform [i,j] = 1/2 * (i,j) - 1/2 * (j,i)
    """
    if len(word) == 2:
        ij = [(word[0],word[1]), 1/2]
        minus_ji = [(word[1],word[0]), -1/2]
        return [ij, minus_ji]
    else:
        return [word, 1]


def signed_two_product(word_1, word_2):
    """
    Perform the product ((i,j) - (j,i)) * ((k,l) - (l,k))
    :param word_1: [tuple of indices, sign]
    :param word_2: [tuple of indices, sign]
    :return: [tuple of indices, sign]
    """
    res = [extract_nested_tuples((word_1[0],word_2[0])), word_1[1]*word_2[1]]
    return res


def signed_n_product(words_1, words_2):
    """
    Product of two list of words
    """
    res = []
    for first_word in words_1:
        for second_word in words_2:
            # print("first_word: ", first_word, "second_word: ", second_word)
            product = signed_two_product(first_word, second_word)
            # print("product: ", product)
            res.append(product)
    return res


def whole_prod(list_of_words):
    """
    Return the product of all the words
    :param list_of_words:
    :return: list_of_words
    """
    whole_prod_res = levy_expand(list_of_words[0])
    for word in list_of_words[1:]:
        expanded_word = levy_expand(word)
        whole_prod_res = signed_n_product(whole_prod_res, expanded_word)
    return whole_prod_res


def get_levy_words(bm_dim):
    """
    Return the list of words representing the levy process
    :param bm_dim: int
    :return: list_of_words
    """
    word_list = []
    for i in range(bm_dim):
        for j in range(i+1, bm_dim):
            word_list.append((i,j))
    return word_list


def nth_moments(bm_dim, n):
    """
    Compute the analytical n-th moments of levy process as T=1
    :param dim:
    :return:
    """
    levy_words = get_levy_words(bm_dim)

    # Compute the expected signature upto degree 8
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sig_length = total_len_algebra(bm_dim, 2*n)
    bm_sig = torch.zeros(1, sig_length).to(device)
    for i in range(bm_dim):
        bm_sig[0][algebra_coordinate_calculator((i, i,), bm_dim)] = 0.5
    expected_sig = transformer(tensor_exp(bm_sig, bm_dim, 2*n))

    res = []
    # Fourth moment of L_{i_1,i_2}, L_{j_1,j_2}, L_{k_1,k_2}, L_{l_1,l_2}
    for n_comb in itertools.combinations_with_replacement(levy_words, r=n):
        n_comb_moment = 0
        # Expand it as linear combinations of elements in the signature
        whole_prod_ = levy_expand(n_comb[0])
        for word in n_comb[1:]:
            expanded_word = levy_expand(word)
            whole_prod_ = signed_n_product(whole_prod_, expanded_word)
        for n_prod_ in whole_prod_:
            sign = n_prod_[1]
            n_prod = n_prod_[0]
            # Express each 4-product of signature elements as a linear combination of higher order terms
            elements_in_degree_2n = shuffle_prod_n_words(n_prod)
            shuffle_result = 0
            # Find the corresponding value in expected signature of BM, perform the linear combination
            for word_2n in elements_in_degree_2n:
                shuffle_result += expected_sig[0, algebra_coordinate_calculator(word_2n, bm_dim)]
            shuffle_result *= sign
            n_comb_moment += shuffle_result
        res.append(n_comb_moment.item())
    nth_moments = torch.tensor(res)
    return nth_moments


if __name__ == '__main__':
    print(nth_moments(4, 4))
