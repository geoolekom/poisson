import numpy as np
from numpy.ma import sin, sqrt, cos

from utils import make_matrix, f


def exact_solution(x, y):
    return sqrt(sin(np.pi * x) * cos(np.pi * y / 2))


def solve_layer(u, n, k):
    if k == 0:
        for t in range(k, n + 1 - k):
            u[t][k] = 0.0
            u[t][n - k] = 0.0
            u[k][t] = sqrt(sin(np.pi * t / n))
            u[n - k][t] = 0.0
    elif k == 1:
        for t in range(k, n + 1 - k):
            a = u[t][k - 1]
            b = u[t - 1][k - 1]
            c = u[t + 1][k - 1]
            u[t][k] = sqrt(- f(t / n, k / n) / n / n + 2 * a ** 2 - b ** 2 / 2 - c ** 2 / 2)

            a = u[t][n - k + 1]
            b = u[t - 1][n - k + 1]
            c = u[t + 1][n - k + 1]
            u[t][n - k] = sqrt(- f(t / n, (n - k) / n) / n / n + 2 * a ** 2 - b ** 2 / 2 - c ** 2 / 2)

            a = u[k - 1][t]
            b = u[k - 1][t - 1]
            c = u[k - 1][t + 1]
            u[k][t] = a + sqrt(- f(k / n, t / n) / n / n + a ** 2 - b ** 2 / 2 - c ** 2 / 2)

            a = u[n - k + 1][t]
            b = u[n - k + 1][t - 1]
            c = u[n - k + 1][t + 1]
            u[n - k][t] = sqrt(- f((n - k) / n, t / n) / n / n + 2 * a ** 2 - b ** 2 / 2 - c ** 2 / 2)
    else:
        for t in range(k, n + 1 - k):
            a = u[t][k - 1]
            b = u[t - 1][k - 1]
            c = u[t + 1][k - 1]
            d = u[t][k - 2]
            u[t][k] = sqrt(- 2 * f(t / n, k / n) / n / n + 4 * a ** 2 - b ** 2 - c ** 2 - d ** 2)

            a = u[t][n - k + 1]
            b = u[t - 1][n - k + 1]
            c = u[t + 1][n - k + 1]
            d = u[t][n - k + 2]
            u[t][n - k] = sqrt(- 2 * f(t / n, (n - k) / n) / n / n + 4 * a ** 2 - b ** 2 - c ** 2 - d ** 2)

            a = u[k - 1][t]
            b = u[k - 1][t - 1]
            c = u[k - 1][t + 1]
            d = u[k - 2][t]
            u[k][t] = sqrt(- 2 * f(k / n, t / n) / n / n + 4 * a ** 2 - b ** 2 - c ** 2 - d ** 2)

            a = u[n - k + 1][t]
            b = u[n - k + 1][t - 1]
            c = u[n - k + 1][t + 1]
            d = u[n - k + 2][t]
            u[n - k][t] = sqrt(- 2 * f((n - k) / n, t / n) / n / n + 4 * a ** 2 - b ** 2 - c ** 2 - d ** 2)


def converge_solution(u, n, *args, **kwargs):
    result = make_matrix(n)
    for t in range(0, n + 1):
        result[t][0] = 0.0
        result[t][n] = 0.0
        result[0][t] = sqrt(sin(np.pi * t / n))
        result[n][t] = 0.0
    for k in range(1, n):
        for m in range(1, n):
            a = u[k - 1][m]
            b = u[k + 1][m]
            c = u[k][m - 1]
            d = u[k][m + 1]
            result[k][m] = sqrt(a * a + b * b + c * c + d * d
                                - 2 * f(k / n, m / n) / n / n) / 2

    diff = np.linalg.norm(result - u, ord='fro')
    return result, diff
