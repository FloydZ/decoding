#!/usr/bin/env python3

from subprocess import Popen, PIPE, STDOUT
from pysat.card import *
import urllib.request
import random
from matrix import *
# from optimize import *


def parse_decodingchallenge(lines): 
    """
    TODO import from optimize.py, but currently I dont want to import CE
    """
    q = 2
    ctr = 1 #f.readline()        # `# n`
    n = int(lines[ctr])
    ctr += 2 # jump over next comment
    tmp = int(lines[ctr]) # `# seed`(GO) or `k`(SD) or # w (QC)
    ctr += 2
    if "vector" in lines[ctr-1]: # QC setting
        k = n // 2
        w = tmp
    else: # GO or SD
        w = int(lines[ctr])
        ctr += 2
        if len(lines[ctr]) <= 3: # ternary
            q = 3
            k = w
            w = int(lines[ctr])
            ctr += 2
        else:
            if tmp == 0: # SD
                k = n // 2
            else: # GO 
                k = tmp
    assert k > 0 
    assert w > 0 
    assert n > 0 
   
    H = []
    while lines[ctr][0] != '#':
        H.append(lines[ctr])
        ctr += 1 
    
    # `# s^transpose` was already consumed
    ctr += 1
    s = lines[ctr].strip("\n")
    H2 = []
    # append the identity matrix.
    for i in range(int(n)-k):
        H2.append("0"*i + "1" + "0"*(int(n)-k-i-1))
    
    for e in H:
        H2.append(e.strip("\n"))
    return n, k, w, q, "".join(H2).strip("'n"), s


def get_decodingchallenge(url: str):
    with urllib.request.urlopen(url) as f:
        lines = f.readlines()
        lines = [line.decode("utf-8").strip("\n") for line in lines]
        return parse_decodingchallenge(lines)


def test_get_decodingchallenge():
    url = "https://decodingchallenge.org/Challenges/SD/SD_100_0"
    print(get_decodingchallenge(url))


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
            if line[0] == "c": 
                continue
            if line [0] == "s":
                assert line == "s SATISFIABLE"
                continue

            line = line[2:]
            s = line.split(" ")
            data += [int(a) for a in s]
       
        for i in data:
            if i >= 1 and i <= n: 
                e.data[0][i-1] = 1
    return e


def check_solution(H: Matrix, S: Matrix, file="out.sol"):
    """
    :return true/false
    """
    e = parse_solution(H.ncols, file)
    s = H*e.transpose()
    s.transpose().print()
    S.print()
    return s == S


def bruteforce(H: Matrix, S: Matrix, file="out.sol"):
    """
    worse then prange
    """
    clauses = [[str(i+1) for i, d in enumerate(row) if d !=0 ] for row in H.data]
    out = ""
    for i, c in enumerate(clauses):
        out += "x" + " ".join(c[:-1])
        if S.data[0][i]: out += " " + c[-1] + " 0\n"
        else: out += " -" + c[-1] + " 0\n"
    
    cnf = CardEnc.atmost(lits=list(range(1, n+1)), bound=w)
    file = "out.cnf"
    cnf.to_file(file)
    with open(file, 'r') as original:
        data = original.read()
    with open(file, 'w') as modified:
        modified.write(out + data)


def random_permutation(H: Matrix, P):
    assert len(P) == H.ncols
    for i in range(len(P)):
        pos = random.randint(i, n-1)
        H.swap_cols(i, pos)
        P[i], P[pos] = P[pos], P[i]


def prange(H: Matrix, S: Matrix, file="out.sol"):
    n = H.ncols
    nk = H.nrows
    P = list(range(n))

    file = "out.cnf"
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


#url = "https://decodingchallenge.org/Challenges/SD/SD_100_0"
url = "https://decodingchallenge.org/Challenges/Goppa/Provider0/old_rng/Goppa_156"
n, k, w, q, H, s = get_decodingchallenge(url)
H = Matrix(n-k, n, 2).from_string(H);
S = Matrix(1, n-k, 2).from_string(s);
#print(n,k,w,q)
#assert(check_solution(H, S, "out.sol"))
prange(H, S)
