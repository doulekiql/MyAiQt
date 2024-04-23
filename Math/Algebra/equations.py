import numpy as np
import matplotlib.pyplot as plt

from vector import Vector
from matrix import Matrix

class Equations:

    def __init__(self, A:Matrix, b:Vector):
        if A.row_len() != len(b):
            raise ValueError("Equation - Matrix A's row number must be equal to the length of Vector b.")

        self._m = A.row_len()
        self._n = A.col_len()
        self._Ab = [Vector(A.row_vector(i).elements() + [b[i]]) for i in range(self._m)]
        self._pivots = []


    def _max_row(self, idx_i, idx_j, n):
        max, idx = self._Ab[idx_i][idx_j], idx_i
        for i in range(idx_i + 1, n):
            if self._Ab[i][idx_j] > max:
                max, idx = self._Ab[i][idx_j], i
        return idx


    def _forward(self):
        i, k = 0, 0
        while i < self._m and k < self._n:
            max_row = self._max_row(i, k, self._m)
            self._Ab[i], self._Ab[max_row] = self._Ab[max_row], self._Ab[i]
            if np.isclose(self._Ab[i][k], 0.0, atol=1e-9):
                k += 1
            else:
                self._Ab[i] = self._Ab[i] / self._Ab[i][k]
                for j in range(i + 1, self._m):
                    self._Ab[j] = self._Ab[j] - self._Ab[j][k] * self._Ab[i]
                self._pivots.append(k)
                i += 1


    def _backward(self):
        n = len(self._pivots)
        for i in range(n - 1, -1, -1):
            k = self._pivots[i]
            for j in range(i - 1, -1, -1):
                self._Ab[j] = self._Ab[j] - self._Ab[j][k] * self._Ab[i]


    def gauss_jordan_elimination(self):
        self._forward()
        self._backward()
        for i in range(len(self._pivots), self._m):
            if not np.isclose(self._Ab[i][-1], 0.0, atol=1e-9):
                return False
        return True

    def print_augmatrix(self):
        for i in range(self._m):
            print(" ".join(str(self._Ab[i][j]) for j in range(self._n)), end=" ")
            print("|", self._Ab[i][-1])


if __name__=="__main__":
    A = Matrix([[2, 3, 1], [4, 7, 8], [9, 6, 7]])
    b = Vector([1, 2, 3])
    eq = Equations(A, b)
    eq.gauss_jordan_elimination()
    eq.print_augmatrix()

    A1 = Matrix([[1, 2, 4], [3, 7, 2], [2, 3, 3]])
    b1 = Vector([7, -11, 1])
    eq1 = Equations(A1, b1)
    eq1.gauss_jordan_elimination()
    eq1.print_augmatrix()
    print()

    A2 = Matrix([[1, -3, 5], [2, -1, -3], [3, 1, 4]])
    b2 = Vector([-9, 19, -13])
    eq2 = Equations(A2, b2)
    eq2.gauss_jordan_elimination()
    eq2.print_augmatrix()
    print()

    A3 = Matrix([[1, 2, -2], [2, -3, 1], [3, -1, 3]])
    b3 = Vector([6, -10, -16])
    eq3 = Equations(A3, b3)
    eq3.gauss_jordan_elimination()
    eq3.print_augmatrix()
    print()

    A4 = Matrix([[3, 1, -2], [5, -3, 10], [7, 4, 16]])
    b4 = Vector([4, 32, 13])
    eq4 = Equations(A4, b4)
    eq4.gauss_jordan_elimination()
    eq4.print_augmatrix()
    print()

    A5 = Matrix([[6, -3, 2], [5, 1, 12], [8, 5, 1]])
    b5 = Vector([31, 36, 11])
    eq5 = Equations(A5, b5)
    eq5.gauss_jordan_elimination()
    eq5.print_augmatrix()
    print()

    A6 = Matrix([[1, 1, 1], [1, -1, -1], [2, 1, 5]])
    b6 = Vector([3, -1, 8])
    eq6 = Equations(A6, b6)
    eq6.gauss_jordan_elimination()
    eq6.print_augmatrix()
    print()

    A7 = Matrix([[1, -1, 2, 0, 3],
                 [-1, 1, 0, 2, -5],
                 [1, -1, 4, 2, 4],
                 [-2, 2, -5, -1, -3]])
    b7 = Vector([1, 5, 13, -1])
    eq7 = Equations(A7, b7)
    eq7.gauss_jordan_elimination()
    eq7.print_augmatrix()
    print()

    A8 = Matrix([[2, 2],
                 [2, 1],
                 [1, 2]])
    b8 = Vector([3, 2.5, 7])
    eq8 = Equations(A8, b8)
    if not eq8.gauss_jordan_elimination():
        print("No Solution!")
    eq8.print_augmatrix()
    print()

    A9 = Matrix([[2, 0, 1],
                 [-1, -1, -2],
                 [-3, 0, 1]])
    b9 = Vector([1, 0, 0])
    eq9 = Equations(A9, b9)
    if not eq9.gauss_jordan_elimination():
        print("No Solution!")
    eq9.print_augmatrix()
    print()