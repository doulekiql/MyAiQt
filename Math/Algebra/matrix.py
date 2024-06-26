import numpy as np
import matplotlib.pyplot as plt

from vector import Vector

class Matrix:

    def __init__(self, lst):
        self._elements = [row[:] for row in lst]

    def __repr__(self):
        return f"Matrix - { self._elements }"


    def __str__(self):
        return f"Matrix - [{ ', '.join(str(e) for e in self._elements) }]"


    def __getitem__(self, pos):
        if isinstance(pos, tuple) or isinstance(pos, list):
            r_pos, c_pos = pos
        else:
            raise TypeError(f"Matrix - unsupported operand type(s) '{type(pos).__name__}', pos must be a tuple.")

        return self._elements[r_pos][c_pos]


    def __eq__(self, another):
        if isinstance(another, Matrix):
            return all([a == b for a, b in zip(self._elements, another._elements)])
        else:
            return False

    def row_vector(self, r_pos):
        return Vector(self._elements[r_pos])


    def col_vector(self, c_pos):
        return Vector([row[c_pos] for row in self._elements])


    def shape(self):
        return len(self._elements), len(self._elements[0])


    def row_len(self):
        return len(self._elements)

    __len__ = row_len

    def col_len(self):
        return len(self._elements[0])


    def size(self):
        return self.row_len() * self.col_len()


    def __add__(self, another):
        if isinstance(another, Matrix):
            if self.shape() != another.shape():
                raise ValueError("Matrix - both matrices must have the same shape.")
            return Matrix([[a + b for a, b in zip(self.row_vector(i), another.row_vector(i))] for i in range(self.row_len())])
        else:
            raise TypeError(f"Matrix - unsupported operand type(s) for: 'Matrix' and '{type(another).__name__}'")


    def __sub__(self, another):
        if isinstance(another, Matrix):
            if self.shape() != another.shape():
                raise ValueError("Matrix - both matrices must have the same shape.")
            return Matrix([[a - b for a, b in zip(self.row_vector(i), another.row_vector(i))] for i in range(self.row_len())])
        else:
            raise TypeError(f"Matrix - unsupported operand type(s) for: 'Matrix' and '{type(another).__name__}'")


    def __mul__(self, k):
        return Matrix([[e * k for e in self.row_vector(i)] for i in range(self.row_len())])


    def __rmul__(self, k):
        return self * k


    def __truediv__(self, k):
        if np.isclose(k, 0.0, atol=1e-9):
            raise ZeroDivisionError("Vector - division by zero")
        else:
            return (1 / k) * self


    def dot(self, another):
        if isinstance(another, Vector):
            if self.col_len() != len(another):
                raise ValueError("Matrix - the number of columns of the matrix must be equal to the dimension of the vector.")
            return Vector([self.row_vector(i).dot(another) for i in range(self.row_len())])
        elif isinstance(another, Matrix):
            if self.shape()[1] != another.shape()[0]:
                raise ValueError("Matrix - the number of columns of the first matrix must be equal to the number of rows of the second matrix.")
            return Matrix([[self.row_vector(i).dot(another.col_vector(j)) for j in range(another.col_len())] for i in range(self.row_len())])
        else:
            raise TypeError(f"Matrix - unsupported operand type(s) for: 'Matrix' and '{type(another).__name__}'")


    def T(self):
        return Matrix([[e for e in self.col_vector(i)] for i in range(self.col_len())])


    @staticmethod
    def identity(n):
        return Matrix([[1 if i == j else 0 for j in range(n)] for i in range(n)])


    @staticmethod
    def zero(r, c):
        return Matrix([[0] * c for _ in range(r)])


def plot(m_p, m_t):
    plt.figure(figsize=(10, 5))
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)

    plt.plot([m_p.col_vector(i)[0] for i in range(m_p.col_len())],
             [m_p.col_vector(i)[1] for i in range(m_p.col_len())])

    m_p = m_t.dot(m_p)
    plt.plot([m_p.col_vector(i)[0] for i in range(m_p.col_len())],
             [m_p.col_vector(i)[1] for i in range(m_p.col_len())])
    plt.show()

if __name__ == "__main__":

    v_1 = Vector([1, 2, 3])
    m_1 = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    m_2 = Matrix([[10, 11, 12], [13, 14, 15], [16, 17, 18]])
    print(f"Vector_1: {v_1}")
    print(f"Matrix_1: {m_1}")
    print(f"Matrix_2: {m_2}")
    print(f"Matrix_1 element (1, 1): {m_1[1, 1]}")
    print(f"Matrix_1 row 1: {m_1.row_vector(1)}")
    print(f"Matrix_1 column 1: {m_1.col_vector(1)}")
    print(f"Matrix_1 shape: {m_1.shape()}")
    print(f"Matrix_1 row length: {m_1.row_len()}")
    print(f"Matrix_1 length: {len(m_1)}")
    print(f"Matrix_1 column length: {m_1.col_len()}")
    print(f"Matrix_1 size: {m_1.size()}")
    print(f"Matrix_1 + Matrix_2: {m_1 + m_2}")
    print(f"Matrix_1 - Matrix_2: {m_1 - m_2}")
    print(f"Matrix_1 * 2: {m_1 * 2}")
    print(f"Matrix_1 / 2: {m_1 / 2}")
    print(f"Matrix_1 dot Vector_1: {m_1.dot(v_1)}")
    print(f"Matrix_1 dot Matrix_2: {m_1.dot(m_2)}")
    print(f"Matrix_2 dot Matrix_1: {m_2.dot(m_1)}")
    print(f"Matrix_1 Transpose: {m_1.T()}")
    print(f"Matrix Identity 3x3: {Matrix.identity(3)}")
    print(f"Matrix Zero 2x3: {Matrix.zero(2, 3)}")

    points = [[0,0], [0,7], [3, 7], [3, 6], [1, 6], [1, 5], [3, 5], [3, 4], [1, 4], [1, 0], [0, 0]]

    print("Transformation: 0.5x")
    plot(Matrix(points).T(), Matrix([[0.5, 0], [0, 0.5]]))

    print("Transformation: 2x")
    plot(Matrix(points).T(), Matrix([[2, 0], [0, 2]]))

    print("Transformation: origin flip")
    plot(Matrix(points).T(), Matrix([[1, 0], [0, -1]]))

    print("Transformation: shear warp")
    plot(Matrix(points).T(), Matrix([[1, 0.5], [0.5, 1]]))

    print("Transformation: rotate 45 degrees")
    plot(Matrix(points).T(), Matrix([[np.cos(np.pi/4), np.sin(np.pi/4)],
                                     [-np.sin(np.pi/4), np.cos(np.pi/4)]]))

    print("Transformation: rotate -90 degrees")
    plot(Matrix(points).T(), Matrix([[0, -1], [1, 0]]))




