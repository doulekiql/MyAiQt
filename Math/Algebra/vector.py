import numpy as np

class Vector:

    def __init__(self, lst):
        self._elements = list(lst)


    def __repr__(self):
        return f"Vector - {self._elements}"


    def __str__(self):
        return f"Vector - [{ ', '.join(str(e) for e in self._elements) }]"


    def __iter__(self):
        return self._elements.__iter__()


    def __getitem__(self, idx):
        return self._elements[idx]


    def __len__(self):
        return len(self._elements)


    def __add__(self, another):
        if isinstance(another, Vector):
            if len(self._elements) != len(another._elements):
                raise ValueError("Vector - both vectors must have the same dimension.")
            return Vector([a + b for a, b in zip(self._elements, another._elements)])
        else:
            raise TypeError(f"Vector - unsupported operand type(s) for: 'Vector' and '{type(another).__name__}'")


    def __sub__(self, other):
        if isinstance(other, Vector):
            if len(self._elements) != len(other._elements):
                raise ValueError("Vector - both vectors must have the same dimension.")
            return Vector([a - b for a, b in zip(self._elements, other._elements)])
        else:
            raise TypeError(f"Vector - unsupported operand type(s) for: 'Vector' and '{type(other).__name__}'")


    def __mul__(self, k):
        return Vector([k * e for e in self._elements])


    def __rmul__(self, k):
        return self * k


    def __pos__(self):
        return 1 * self


    def __neg__(self):
        return -1 * self


    def __truediv__(self, k):
        if np.isclose(k, 0.0, atol=1e-9):
            raise ZeroDivisionError("Vector - division by zero")
        else:
            return (1 / k) * self


    def dot(self, another):
        if isinstance(another, Vector):
            if len(self._elements) != len(another._elements):
                raise ValueError("Vector - both vectors must have the same dimension.")
            return sum(a * b for a, b in zip(self, another))
        else:
            raise TypeError(f"Vector - unsupported operand type(s) for: 'Vector' and '{type(another).__name__}'")


    def magnitude(self):
        return np.sqrt(sum(e ** 2 for e in self._elements))


    def normalize(self):
        magnitude = self.magnitude()
        if np.isclose(magnitude, 0.0, atol=1e-9):
            raise ZeroDivisionError("Vector - magnitude is zero")
        return Vector(self._elements) / self.magnitude()


    def elements(self):
        return self._elements

    @classmethod
    def zero(cls, dim):
        return Vector([0] * dim)


if __name__=="__main__":
    v_1 = Vector([1, 2])
    v_2 = Vector([3, 4])
    print(f"Vector_1: {v_1}" )
    print(f"Vector_1 length: {len(v_1)}")
    print(f"Vector_2: {v_2}")
    print(f"Vector_2 length: {len(v_2)}")
    print(f"Vector_1 scalar multiplication by 3: {v_1 * 3}")
    print(f"Vector_1 positive: {+v_1}")
    print(f"Vector_1 negative: {-v_1}")
    print(f"Vector_2 scalar multiplication by 3: {3 * v_2}")
    print(f"Vector_2 division: {v_2 / 2}")
    print(f"Vector_1 + Vector_2: {v_1 + v_2}")
    print(f"Vector_1 - Vector_2: {v_1 - v_2}")
    print(f"Vector_1 dot Vector_2: {v_1.dot(v_2)}")
    print(f"Vector_1 magnitude: {v_1.magnitude()}")
    print(f"Vector_1 normalize: {v_1.normalize()}")
    print(f"Zero Vector: {Vector.zero(5)}")
