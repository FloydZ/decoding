#!/usr/bin/env python3
""" tests for the python package """

from decoding.matrix import Matrix
from decoding.challenge import parse_decodingchallenge


def test_matrix_gauss():
    A = Matrix(3, 3, 2)
    A.from_string("110011101", transposed=False)
    assert A.gauß() == 2


def test_matrix_mul():
    A = Matrix(2, 2, 3).from_string("1201", transposed=False)
    B = Matrix(2, 1, 3).from_string("12", transposed=False)
    C = A * B
    assert C[0, 0] == 2 and C[1, 0] == 2


def test_parse_decodingchallenge():
    # SD instance: n, seed, w, H^T, s^T
    lines = ["# n", "8", "# seed", "0", "# w", "2",
             "# H^transpose (each line corresponds to column of H, the identity part is omitted)",
             "1100", "0110", "0011", "1001", "# s^transpose", "1010"]
    n, k, w, q, H, s = parse_decodingchallenge(lines)
    assert (n, k, w, q) == (8, 4, 2, 2)
    assert H == "1000010000100001" + "1100011000111001"
    assert s == "1010"
