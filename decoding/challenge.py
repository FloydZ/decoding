#!/usr/bin/env python3
""" parser for the instances of https://decodingchallenge.org """

import urllib.request


def parse_decodingchallenge(lines):
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
    return n, k, w, q, "".join(H2).strip("\n"), s


def get_decodingchallenge(url: str):
    with urllib.request.urlopen(url) as f:
        lines = f.readlines()
        lines = [line.decode("utf-8").strip("\n") for line in lines]
        return parse_decodingchallenge(lines)


def test_get_decodingchallenge():
    url = "https://decodingchallenge.org/Challenges/SD/SD_100_0"
    print(get_decodingchallenge(url))
