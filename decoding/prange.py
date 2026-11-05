#!/usr/bin/env python3

from subprocess import Popen, PIPE, STDOUT
from pysat.card import *
import random
try:
    from .matrix import Matrix
    from .challenge import get_decodingchallenge
except ImportError:  # executed as a script
    from matrix import Matrix
    from challenge import get_decodingchallenge
# from optimize import *


def run_cryptominisat(file: str = "out.cnf", seconds: int=30, sol: str = "out.sol"):
    """
    writes the output of cryptominisat into `sol`
    :return true/false if found or not
    """
    cmd = ["cryptominisat5", "--maxtime", str(seconds), file]
    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    # NOTE: `communicate` instead of `wait`, which can deadlock on a full pipe
    out, _ = p.communicate()
    with open(sol, "w") as f:
        f.write(out.decode("utf-8"))
    # NOTE: the exit code is 10 for SAT, 20 for UNSAT and 15 for unknown
    return p.returncode == 10


def parse_solution(n: int, file="out.sol") -> Matrix:
    """ parses the output of cryptominisat. Skips over all `c` comment lines.
    Checks if the output is `SATISFIABLE` else asserts
    Reads the first n variables from `v` and returns them as a row vector
    """
    e = Matrix(1, n)
    with open(file, 'r') as f:
        data = []
        for line in f.readlines():
            line = line.strip()
            if not line or line[0] == "c":
                continue
            if line[0] == "s":
                # NOTE: `readlines` keeps the trailing newline
                assert line == "s SATISFIABLE"
                continue

            line = line[2:]
            data += [int(a) for a in line.split()]
       
        for i in data:
            if i >= 1 and i <= n: 
                e.data[0][i-1] = 1
    return e


def check_solution(H: Matrix, S: Matrix, file="out.sol"):
    """
    :return true/false
    """
    e = parse_solution(H.ncols, file)
    s = (H*e.transpose()).transpose()
    s.print()
    S.print()
    # NOTE: `Matrix` has no `__eq__`, so `==` compared object identities
    return s.data == S.data


def bruteforce(H: Matrix, S: Matrix, w: int, file="out.cnf"):
    """
    worse then prange
    """
    n = H.ncols
    clauses = [[str(i+1) for i, d in enumerate(row) if d !=0 ] for row in H.data]
    out = ""
    for i, c in enumerate(clauses):
        out += "x" + " ".join(c[:-1])
        if S.data[0][i]: out += " " + c[-1] + " 0\n"
        else: out += " -" + c[-1] + " 0\n"
    
    cnf = CardEnc.atmost(lits=list(range(1, n+1)), bound=w)
    cnf.to_file(file)
    with open(file, 'r') as original:
        data = original.read()
    with open(file, 'w') as modified:
        modified.write(out + data)


def random_permutation(H: Matrix, P):
    assert len(P) == H.ncols
    for i in range(len(P)):
        pos = random.randint(i, len(P)-1)
        H.swap_cols(i, pos)
        P[i], P[pos] = P[pos], P[i]


def prange(H: Matrix, S: Matrix, w: int, file="out.cnf", sol="out.sol"):
    """
    NOTE: permutes the columns of `H`
    :return the error vector e with H*e^T = S^T
    """
    n = H.ncols
    nk = H.nrows
    k = n - nk
    P = list(range(n))

    seconds = 30
    while 1:
        random_permutation(H, P)

        # NOTE: the syndrome must be transformed as well, hence `[H | S^T]`
        A = Matrix(nk, n + 1, H.q)
        for i in range(nk):
            A.data[i] = H.data[i] + [S.data[0][i]]
        if A.gauß(nk) < nk:
            continue

        # A = [I | H' | S'], every row is a xor clause on e2: H'_i * e2 = S'_i
        out, nr_clauses, solvable = "", 0, True
        for row in A.data:
            lits = [str(i+1) for i in range(k) if row[nk + i] != 0]
            if not lits:
                # 0 = S'_i
                solvable &= row[n] == 0
                continue
            # NOTE: negating a literal flips the parity
            if row[n] == 0:
                lits[-1] = "-" + lits[-1]
            out += "x" + " ".join(lits) + " 0\n"
            nr_clauses += 1
        if not solvable:
            continue

        cnf = CardEnc.atmost(lits=list(range(1, k+1)), bound=w)
        with open(file, 'w') as f:
            f.write("p cnf {} {}\n".format(max(cnf.nv, k), nr_clauses + len(cnf.clauses)))
            f.write(out)
            for c in cnf.clauses:
                f.write(" ".join(str(a) for a in c) + " 0\n")

        if run_cryptominisat(file, seconds, sol):
            # reconstruct: e = (0 | e2), then undo the permutation
            e2 = parse_solution(k, sol)
            e = Matrix(1, n, H.q)
            for i in range(k):
                e.data[0][P[nk + i]] = e2.data[0][i]
            return e


if __name__ == "__main__":
    #url = "https://decodingchallenge.org/Challenges/SD/SD_100_0"
    url = "https://decodingchallenge.org/Challenges/Goppa/Provider0/old_rng/Goppa_156"
    n, k, w, q, H, s = get_decodingchallenge(url)
    H = Matrix(n-k, n, 2).from_string(H)
    S = Matrix(1, n-k, 2).from_string(s)
    #print(n,k,w,q)
    #assert(check_solution(H, S, "out.sol"))
    # NOTE: `prange` permutes the columns of its argument
    e = prange(H.copy(), S, w)
    e.print()
    s = (H*e.transpose()).transpose()
    assert s.data == S.data
