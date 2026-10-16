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


def run_cryptominisat(file: str = "out.cnf", seconds: int=30):
    """
    :return true/false if found or not
    """
    cmd = ["cryptominisat5", "--maxtime", str(seconds), file] 
    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    p.wait()
    s = str(p.stdout.read())
    print(s)
    return p.returncode == 0


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


def prange(H: Matrix, S: Matrix, w: int, file="out.cnf"):
    n = H.ncols
    nk = H.nrows
    k = n - nk
    P = list(range(n))

    seconds = 30
    while 1:
        random_permutation(H, P)
        H.gauß()
        #H.print()
        
        clauses = [[str(i+1) for i in range(k) if row[nk + i] !=0 ] for row in H.data]
        out = ""
        for i, c in enumerate(clauses):
            out += "x" + " ".join(c[:-1])
            if S.data[0][i]: out += " " + c[-1] + " 0\n"
            else: out += " -" + c[-1] + " 0\n"
        
        
        cnf = CardEnc.atmost(lits=list(range(1, k+1)), bound=w)
        cnf.to_file(file)
        with open(file, 'r') as original:
            data = original.read()
        with open(file, 'w') as modified:
            modified.write(out + data)

        run_cryptominisat(file, seconds)

    # TODO reconstruct


if __name__ == "__main__":
    #url = "https://decodingchallenge.org/Challenges/SD/SD_100_0"
    url = "https://decodingchallenge.org/Challenges/Goppa/Provider0/old_rng/Goppa_156"
    n, k, w, q, H, s = get_decodingchallenge(url)
    H = Matrix(n-k, n, 2).from_string(H)
    S = Matrix(1, n-k, 2).from_string(s)
    #print(n,k,w,q)
    #assert(check_solution(H, S, "out.sol"))
    prange(H, S, w)
