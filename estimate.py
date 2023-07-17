#from sage.all import *
from math import comb as binom
from math import log2, inf
from math import *
import math


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


def _optimize_m4ri(n: int, k: int, mem=inf):
    """
    Find optimal blocksize for Gaussian elimination via M4RI
    INPUT:
    - ``n`` -- Row additons are perfomed on ``n`` coordinates
    - ``k`` -- Matrix consists of ``n-k`` rows
    """

    (r, v) = (0, inf)
    for i in range(n - k):
        tmp = float(_gaussian_elimination_complexity(n, k, i))
        tmp = math.log2(tmp)
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


def _list_merge_async_complexity(L1,L2, l, hmap):
    """
    Complexity estimate of merging two lists exact

    INPUT:

    - ``L`` -- size of lists to be merged
    - ``l`` -- amount of bits used for matching
    - ``hmap`` -- indicates if hashmap is being used (Default 0: no hashmap)

    EXAMPLES::

        sage: from tii.asymmetric_ciphers.cbc.complexities.syndrome_decoding.binary_estimator import _list_merge_complexity
        sage: _list_merge_complexity(L=2**16,l=16,hmap=1) # random

    """

    if L1 == 1 and L2==1:
        return 1
    if L1==1:
        return L2
    if L2==1:
        return L1
    if not hmap:
        return 0 #to be implemented
    else:
        return L1+L2 + L1*L2 // 2 ** l


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
        return max(1, 2 * int(math.log2(L)) * L + L ** 2 // 2 ** l)
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
        lam = max(0, int(min(math.ceil(math.log2(L)), l - 2 * w)))
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

    solutions = max(0, math.log2(binom(n, w)) - (n - k))

    r = _optimize_m4ri(n, k, mem)
    Tp = max(math.log2(binom(n, w)) - log2(binom(n - k, w)) - solutions, 0)
    Tg = log2(_gaussian_elimination_complexity(n, k, r))
    time = Tp + Tg
    memory = log2(_mem_matrix(n, k, r))

    time += __memory_access_cost(memory, memory_access)

    params = [r]

    par = {"r": params[0]}
    res = {"time": time, "memory": memory, "parameters": par}
    return res


def marcovchain_number_perms(n: int, k: int, w: int, c: int, p: int, l: int):
    """
    :param n: code length
    :param k: code dimension
    :param w: weight
    :param c: number of coordinates to exchange during each gaussian elimination
    :param p: weight of the good state
    :param l: window
    Example:

        p = 3
        n = 2918
        k = n//2
        w = 56
        l = 0
        c = 95
        number_perms(n,k,w,c,p,l)
    :return: number of expected iteration in logarithmic notation
    """
    R1 = RealField(150)

    def transition(u: int, d: int, c: int, n: int, k: int, w: int, l: int):
        # from u to u+d by exchanging c columns
        return R1(sum(binom(w-u, i) * binom(n-k-l-w+u, c-i) * binom(u, i-d) * binom(k+l-u, c+d-i)
                      for i in range(max(d, 0), min(w-u+1, c+1, c+d+1))))/R1(binom(n-k-l, c) * binom(k+l, c))
    A = matrix(R1, w+1, w+1)
    for i in range(w+1):
        for j in range(w+1):
            A[i,j] = transition(i, j-i, c, n, k, w, l)

    # transition matrix excluding success-state
    B = A[[i for i in range(w+1) if i!=p],[i for i in range(w+1) if i!=p]]

    # fundamental matrix of markov process
    R = (identity_matrix(R1, w, w)-B)**(-1)

    # initial state of markov chain
    state = [(binom(n-k,w-i)*binom(k,i))/binom(n,w) for i in range(w+1) if i!=p]

    # number of permutations
    return math.log2(sum(state[i]*sum(R[i,j] for j in range(w)) for i in range(w))) - log2(n-k)

