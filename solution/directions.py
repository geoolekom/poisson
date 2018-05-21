import numpy as np
from utils import make_matrix, f, tdma


def iteration(u, n, t):
    result = make_matrix(n)
    for k in range(0, n + 1):
        result[0][k] = u[0][k]
        result[n][k] = u[n][k]

    a = [0] + [- t * n * n] * (n - 1) + [0]
    b = [1] + [1 + 2 * t * n * n] * (n - 1) + [1]
    c = [0] + [- t * n * n] * (n - 1) + [0]
    for k in range(1, n):
        d = u[k] + t * (u[k - 1] + u[k + 1] - 2 * u[k]) * n * n - t * f(k / n, np.linspace(0, 1, n + 1))
        d[0] = u[k][0]
        d[n] = u[k][n]
        sol = tdma(n + 1, a, b, c, d)
        result[k] = sol
    return result


def converge_solution(u, n, *args, **kwargs):
    t = 1/(2 * np.pi * n)
    step1 = iteration(u, n, t)
    step2 = iteration(np.transpose(step1), n, t)
    result = np.transpose(step2)
    diff = np.linalg.norm(result - u, ord='fro')
    return result, diff
