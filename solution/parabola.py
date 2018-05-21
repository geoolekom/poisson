import numpy as np

from utils import make_matrix, f


def converge_solution(u, n, *args, **kwargs):
    t = 1 / 100000
    result = make_matrix(n)
    for k in range(0, n + 1):
        result[k][0] = u[k][0]
        result[k][n] = u[k][n]
        result[0][k] = u[0][k]
        result[n][k] = u[n][k]
    for k in range(1, n):
        for m in range(1, n):
            a = u[k - 1][m]
            b = u[k + 1][m]
            c = u[k][m - 1]
            d = u[k][m + 1]
            e = u[k][m]
            delta = (a * a + b * b + c * c + d * d - 4 * e * e) * n * n / 2 - f(k / n, m / n)
            result[k][m] = e + delta * t
    diff = np.linalg.norm(result - u, ord='fro')
    return result, diff
