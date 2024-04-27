from matrix import Matrix

class Decomposition:

    def LU(self, A:Matrix):
        if A.row_len() != A.col_len():
            raise ValueError("Matrix - the number of rows must be equal to the number of columns.")

        n = A.row_len()
        L = Matrix.identity(n)
        U = Matrix.zero(n, n)

        for i in range(n):
            for j in range(i, n):
                U[i, j] = A[i, j]
                for k in range(i):
                    U[i, j] -= L[i, k] * U[k, j]
            for j in range(i + 1, n):
                L[j, i] = A[j, i]
                for k in range(i):
                    L[j, i] -= L[j, k] * U[k, i]
                L[j, i] /= U[i, i]

        return L, U


    def PLU(self, A:Matrix):
        if A.row_len() != A.col_len():
            raise ValueError("Matrix - the number of rows must be equal to the number of columns.")

        n = A.row_len()
        L = Matrix.identity(n)
        U = Matrix.zero(n, n)
        P = Matrix.identity(n)

        for i in range(n):
            max_row = i
            for j in range(i + 1, n):
                if abs(A[j, i]) > abs(A[max_row, i]):
                    max_row = j
            A[i], A[max_row] = A[max_row], A[i]
            P[i], P[max_row] = P[max_row], P[i]
            for j in range(i, n):
                U[i, j] = A[i, j]
                for k in range(i):
                    U[i, j] -= L[i, k] * U[k, j]
            for j in range(i + 1, n):
                L[j, i] = A[j, i]
                for k in range(i):
                    L[j, i] -= L[j, k] * U[k, i]
                L[j, i] /= U[i, i]

        return P, L, U
