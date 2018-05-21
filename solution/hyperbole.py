import numpy as np
from utils import make_matrix, f, tdma, lambda_xx, lambda_yy, delta


def iteration(u, n, t, prev):
    s = 0.5
    g = n / 2
    result = np.zeros((n + 1, n + 1))

    delta_u = delta(u, n)
    a = [0] + [- t * t * s * n * n] * (n - 1) + [0]
    b = [1] + [1 + 2 * t * t * s * n * n] * (n - 1) + [1]
    c = [0] + [- t * t * s * n * n] * (n - 1) + [0]
    temp = make_matrix(n)
    for k in range(1, n):
        d = - (u[k] - prev[k]) * t * g + t * t * (delta_u[k] - f(k / n, np.linspace(0, 1, n + 1)) / n / n) * n * n
        d[0] = 0
        d[n] = 0
        temp[k] = tdma(n + 1, a, b, c, d)

    temp = np.transpose(temp)
    for k in range(1, n):
        d = temp[k]
        d[0] = 0
        d[n] = 0
        result[k] = tdma(n + 1, a, b, c, d)

    result = np.transpose(result)
    return result + 2 * u - prev


def converge_solution(u, n, prev=None, *args, **kwargs):
    t = 1 / 1000
    g = n / 2
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
            result[k][m] = 2 * e - prev[k][m] \
                - g * t * (e - prev[k][m]) + t * t * \
                ((a * a + b * b + c * c + d * d - 4 * e * e) * n * n / 2
                - f(k / n, m / n))

            # print(f'{k},{m}\t\t', result[k][m], e)
    diff = np.linalg.norm(result - u, ord='fro')
    return result, diff, u
