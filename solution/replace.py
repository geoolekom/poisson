import numpy as np
from numpy.ma import sin, cos

from utils import make_matrix, f


def exact_solution(x, y):
    return 0.5 * sin(np.pi * x) * cos(np.pi * y / 2)


def solve_layer(u, n, k):
    if k == 0:
        for t in range(k, n + 1 - k):
            u[t][k] = 0.0
            u[t][n - k] = 0.0
            u[k][t] = 0.5 * sin(np.pi * t / n)
            u[n - k][t] = 0.0
    elif k == 1:
        for t in range(k, n + 1 - k):
            a = u[t][k - 1]
            b = u[t - 1][k - 1]
            c = u[t + 1][k - 1]
            u[t][k] = f(t / n, (k - 1) / n) / n / n + 3 * a - b - c

            a = u[t][n - k + 1]
            b = u[t - 1][n - k + 1]
            c = u[t + 1][n - k + 1]
            u[t][n - k] = f(t / n, (n - k + 1) / n) / n / n + 3 * a - b - c

            a = u[k - 1][t]
            b = u[k - 1][t - 1]
            c = u[k - 1][t + 1]
            u[k][t] = f((k - 1) / n, t / n) / n / n + 3 * a - b - c

            a = u[n - k + 1][t]
            b = u[n - k + 1][t - 1]
            c = u[n - k + 1][t + 1]
            u[n - k][t] = f((n - k + 1) / n, t / n) / n / n + 3 * a - b - c
    else:
        for t in range(k, n + 1 - k):
            a = u[t][k - 1]
            b = u[t - 1][k - 1]
            c = u[t + 1][k - 1]
            d = u[t][k - 2]
            u[t][k] = f(t / n, (k - 1) / n) / n / n + 4 * a - b - c - d

            a = u[t][n - k + 1]
            b = u[t - 1][n - k + 1]
            c = u[t + 1][n - k + 1]
            d = u[t][n - k + 2]
            u[t][n - k] = f(t / n, (n - k + 1) / n) / n / n + 4 * a - b - c - d

            a = u[k - 1][t]
            b = u[k - 1][t - 1]
            c = u[k - 1][t + 1]
            d = u[k - 2][t]
            u[k][t] = f((k - 1) / n, t / n) / n / n + 4 * a - b - c - d

            a = u[n - k + 1][t]
            b = u[n - k + 1][t - 1]
            c = u[n - k + 1][t + 1]
            d = u[n - k + 2][t]
            u[n - k][t] = f((n - k + 1) / n, t / n) / n / n + 4 * a - b - c - d


def converge_solution(u, n):
    result = make_matrix(n)
    diff = 0
    for t in range(0, n + 1):
        result[t][0] = 0.0
        result[t][n] = 0.0
        result[0][t] = 0.5 * sin(np.pi * t / n)
        result[n][t] = 0.0
    for k in range(1, n):
        for m in range(1, n):
            a = u[k - 1][m]
            b = u[k + 1][m]
            c = u[k][m - 1]
            d = u[k][m + 1]
            result[k][m] = 0.25 * (a + b + c + d - f(k / n, m / n) / n / n)
            diff += (result[k][m] - u[k][m] or 0) ** 2
    return result, diff


