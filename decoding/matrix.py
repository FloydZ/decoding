#!/usr/bin/env python3
""" super simple matrix implementation. The only goal is to have zero dependencies """

from typing import Union
import random



class Matrix:
    """ simple matrix class """
    def __init__(self, nrows: int, ncols: int, q: int = 2) -> None:
        """ zero initialized """
        self.nrows = nrows
        self.ncols = ncols
        self.q = q
        self.data = [[0 for _ in range(ncols)] for _ in range(nrows)] 

    def __getitem__(self, tup):
        """ nice access function """
        x, y = tup
        assert x < self.nrows and y < self.ncols
        return self.data[x][y]

    def print(self, tranpose: bool = False):
        """ printing """
        for i in range(self.nrows):
            for j in range(self.ncols):
                print(self.data[i][j], end='')

            print("")
    
    def zero(self) -> 'Matrix':
        """ zeros all elements"""
        for i in range(self.nrows):
            for j in range(self.ncols):
                self.data[i][j] = 0
        return self

    def random(self) -> 'Matrix':
        """ generates a random matrix """
        for i in range(self.nrows):
            for j in range(self.ncols):
                self.data[i][j] = random.randint(0, self.q - 1)
        return self

    def random_row_with_weight(self, row: int, w: int) -> 'Matrix':
        """ generates a random weight w row """
        assert w > 0 and w < self.ncols
        self.zero()
        for i in range(w):
            self.data[row][i] = 1

        # and now just simple apply a random permutation
        for i in range(self.ncols):
            pos = random.randint(0, self.ncols - i - 1)
            tmp = self.data[row][i]
            self.data[row][i] = self.data[row][i + pos]
            self.data[row][i + pos] = tmp
        return self

    def gauß(self, max_rank: Union[int, None] = None) -> int:
        """ simple Gaussian elimination. Is an inplace operation
        :return the rank of the matrix
        """
        if max_rank is None:
            max_rank = self.nrows
        
        assert isinstance(max_rank, int)
        row = 0
        for col in range(self.ncols):
            if row >= min(max_rank, self.nrows): break

            # find pivot
            sel = -1
            for i in range(row, self.nrows):
                if self.data[i][col] == 1:
                    sel = i 
                    break

            if sel == -1:
                return row

            self.__swap_rows(sel, row)

            # solve remaining coordinates
            for i in range(self.nrows):
                if i == row: continue 
                if self.data[i][col] == 0: continue

                for j in range(self.ncols):
                    self.data[i][j] += self.data[row][j]
                    self.data[i][j] %= self.q

            row += 1
        
        return row

    def mul(self, B: 'Matrix') -> 'Matrix':
        """ simple multiplication """
        B_r, B_c = B.nrows, B.ncols
        assert self.q == B.q and self.ncols == B_r
        C = Matrix(self.nrows, B_c, self.q)
        for i in range(B_c):  # each column in B
            for j in range(self.nrows):  # each row in A
                sum = 0
                for k in range(self.ncols):  # each element in a row in A
                    sum += self[j, k] * B[k, i]

                C.data[j][i] = sum % self.q
        return C

    def add(self, B: 'Matrix') -> 'Matrix':
        """ simple inplace additions """
        B_r, B_c = B.nrows, B.ncols
        assert self.q == B.q and self.ncols == B_c and self.nrows == B_r
        for i in range(self.nrows):
            for j in range(self.ncols):
                self.data[i][j] += B[i, j]
                self.data[i][j] %= self.q
        return self

    def transpose(self) -> 'Matrix':
        """ simple transpose """
        T = Matrix(self.ncols, self.nrows, q=self.q)
        
        for i in range(self.nrows):
            for j in range(self.ncols):
                T.data[j][i] = self.data[i][j]
        return T

    def popcnt_row(self, row: int) -> int:
        """ computes the hamming weight of a row"""
        assert row < self.nrows
        return sum(self.data[row])
        
    def popcnt_col(self, col: int) -> int:
        """ computes the hamming weight of a column"""
        assert col < self.ncols
        t = 0
        for j in range(self.nrows):
            t += self.data[j][col]
        return t

    def __swap_rows(self, i: int, j: int) -> None:
        """ swap the rows i and j """
        assert i < self.nrows and j < self.nrows
        if i == j: return
        for k in range(self.ncols):
            tmp = self.data[i][k]
            self.data[i][k] = self.data[j][k]
            self.data[j][k] = tmp

    def __swap_cols(self, i: int, j: int) -> None:
        """ swap the cols i and j """
        assert i < self.ncols and j < self.ncols
        if i == j: return
        for k in range(self.nrows):
            tmp = self.data[k][i]
            self.data[k][i] = self.data[k][j]
            self.data[k][j] = tmp


if __name__ == "__main__":
    nc, nr, q, w = 10, 5, 2, 2
    A = Matrix(nr, nc, q)
    A.print()
    print()

    A.random()
    A.print()
    print()

    rank = A.gauß()
    A.print()
    print("rank", rank)

    e = Matrix(1, nc, q)
    e.random_row_with_weight(0, w)
    e.print()
    print()

    eT = e.transpose()
    eT.print()
    print()

    C = A.mul(eT)
    C.print()

