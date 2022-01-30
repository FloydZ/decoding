import argparse
import json
import os
import random
import string
from subprocess import Popen, PIPE, STDOUT

import math
import scipy.special as ss
import time

try:
    from math import comb as binom
except:
    pass

from math import log2,inf
from math import *


def __truncate(x, precision):
    """
    Truncates a float
    INPUT:
    - ``x`` -- value to be truncated
    - ``precision`` -- number of decimal places to after which the ``x`` is truncated
    """

    return float(int(x * 10 ** precision) / 10 ** precision)


def __concat_pretty_tables(t1, t2):
    v = t1.split("\n")
    v2 = t2.split("\n")
    vnew = ""
    for i in range(len(v)):
        vnew += v[i] + v2[i][1:] + "\n"
    return vnew[:-1]


def __round_or_truncate_to_given_precision(T, M, truncate, precision):
    if truncate:
        T, M = __truncate(T, precision), __truncate(M, precision)
    else:
        T, M = round(T, precision), round(M, precision)
    return '{:.{p}f}'.format(T, p=precision), '{:.{p}f}'.format(M, p=precision)


def __memory_access_cost(mem, memory_access):
    if memory_access == 0:
        return 0
    elif memory_access == 1:
        return log2(mem)
    elif memory_access == 2:
        return mem / 2
    elif memory_access == 3:
        return mem / 3
    elif callable(memory_access):
        return memory_access(mem)
    return 0


def _gaussian_elimination_complexity(n, k, r):
    """
    Complexity estimate of Gaussian elimination routine
    INPUT:
    - ``n`` -- Row additons are perfomed on ``n`` coordinates
    - ``k`` -- Matrix consists of ``n-k`` rows
    - ``r`` -- Blocksize of method of the four russian for inversion, default is zero
    [Bar07]_ Bard, G.V.: Algorithms for solving linear and polynomial systems of equations over finite fields
    with applications to cryptanalysis. Ph.D. thesis (2007)
    [BLP08] Bernstein, D.J., Lange, T., Peters, C.: Attacking and defending the mceliece cryptosystem.
    In: International Workshop on Post-Quantum Cryptography. pp. 31–46. Springer (2008)
    """

    if r != 0:
        return (r ** 2 + 2 ** r + (n - k - r)) * int(((n + r - 1) / r))

    return (n - k) ** 2


def _optimize_m4ri(n, k, mem=inf):
    """
    Find optimal blocksize for Gaussian elimination via M4RI
    INPUT:
    - ``n`` -- Row additons are perfomed on ``n`` coordinates
    - ``k`` -- Matrix consists of ``n-k`` rows
    """

    (r, v) = (0, inf)
    for i in range(n - k):
        tmp = log2(_gaussian_elimination_complexity(n, k, i))
        if v > tmp and r < mem:
            r = i
            v = tmp
    return r


def _mem_matrix(n, k, r):
    """
    Memory usage of parity check matrix in vector space elements
    INPUT:
    - ``n`` -- length of the code
    - ``k`` -- dimension of the code
    - ``r`` -- block size of M4RI procedure
    """
    return n - k + 2 ** r


def _list_merge_complexity(L, l, hmap):
    """
    Complexity estimate of merging two lists exact
    INPUT:
    - ``L`` -- size of lists to be merged
    - ``l`` -- amount of bits used for matching
    - ``hmap`` -- indicates if hashmap is being used (Default 0: no hashmap)
    """

    if L == 1:
        return 1
    if not hmap:
        return max(1, 2 * int(log2(L)) * L + L ** 2 // 2 ** l)
    else:
        return 2 * L + L ** 2 // 2 ** l


def _indyk_motwani_complexity(L, l, w, hmap, lam=0):
    """
    Complexity of Indyk-Motwani nearest neighbor search

    INPUT:

    - ``L`` -- size of lists to be matched
    - ``l`` -- amount of bits used for matching
    - ``w`` -- target weight
    - ``hmap`` -- indicates if hashmap is being used (Default 0: no hashmap)

    EXAMPLES::
    """

    if w == 0:
        return _list_merge_complexity(L, l, hmap)
    if lam==0 or lam>l-w:
        lam = max(0, int(min(ceil(log2(L)), l - 2 * w)))
    return binom(l, lam) // binom(l - w, lam) * _list_merge_complexity(L, lam, hmap)


def _mitm_nn_complexity(L, l, w, hmap):
    """
    Complexity of Indyk-Motwani nearest neighbor search
    INPUT:
    - ``L`` -- size of lists to be matched
    - ``l`` -- amount of bits used for matching
    - ``w`` -- target weight
    - ``hmap`` -- indicates if hashmap is being used (Default 0: no hashmap)
    """
    if w == 0:
        return _list_merge_complexity(L, l, hmap)
    L1 = L * binom(l / 2, w / 2)
    return _list_merge_complexity(L1, l, hmap)


def prange_complexity(n, k, w, mem=inf, memory_access=0):
    """
    Complexity estimate of Prange's ISD algorithm
    [Pra62] Prange, E.: The use of information sets in decoding cyclic codes. IRE Transactions
                        on Information Theory 8(5), 5–9 (1962)
    expected weight distribution::
        +--------------------------------+-------------------------------+
        | <----------+ n - k +---------> | <----------+ k +------------> |
        |                w               |              0                |
        +--------------------------------+-------------------------------+
    INPUT:
    - ``n`` -- length of the code
    - ``k`` -- dimension of the code
    - ``w`` -- Hamming weight of error vector
    - ``mem`` -- upper bound on the available memory (as log2(bits)), default unlimited
    - ``memory_access`` -- specifies the memory access cost model (default: 0, choices: 0 - constant, 1 - logarithmic, 2 - square-root, 3 - cube-root or deploy custom function which takes as input the logarithm of the total memory usage)
    """

    solutions = max(0, log2(binom(n, w)) - (n - k))

    r = _optimize_m4ri(n, k, mem)
    Tp = max(log2(binom(n, w)) - log2(binom(n - k, w)) - solutions, 0)
    Tg = log2(_gaussian_elimination_complexity(n, k, r))
    time = Tp + Tg
    memory = log2(_mem_matrix(n, k, r))

    time += __memory_access_cost(memory, memory_access)

    params = [r]

    par = {"r": params[0]}
    res = {"time": time, "memory": memory, "parameters": par}
    return res
