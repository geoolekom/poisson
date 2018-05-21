import numpy as np


def make_matrix(n):
    return np.empty([n + 1, n + 1], dtype=object)


def f(y, x):
    return - np.sin(np.pi * x) * np.cos(np.pi * y / 2) * 5 * np.pi * np.pi / 8


def tdma(n, a, b, c, d):
    alpha, beta, x = map(np.zeros, (n + 1, n + 1, n))

    alpha[0] = c[0]/b[0]
    beta[0] = d[0]/b[0]

    for k in range(n):
        denom = b[k] - alpha[k] * a[k]
        alpha[k + 1] = c[k]/denom
        beta[k + 1] = (d[k] - beta[k] * a[k])/denom

    x[n - 1] = beta[n]

    for k in range(n - 2, -1, -1):
        x[k] = beta[k + 1] - alpha[k + 1] * x[k + 1]

    del alpha, beta
    return x


def lambda_yy(matrix, n):
    result = make_matrix(n)
    result[0] = np.zeros(n + 1)
    result[n] = np.zeros(n + 1)
    for k in range(1, n):
        result[k] = n * n * (matrix[k + 1] + matrix[k - 1] - 2 * matrix[k])
    return result


def lambda_xx(matrix, n):
    t = np.transpose(matrix)
    result_t = lambda_yy(t, n)
    return np.transpose(result_t)


def delta(matrix, n):
    result = make_matrix(n)
    result[0] = np.zeros(n + 1)
    result[n] = np.zeros(n + 1)
    for k in range(1, n):
        result[k][0] = 0
        result[k][n] = 0
        for m in range(1, n):
            result[k][m] = matrix[k + 1][m] + matrix[k - 1][m] + matrix[k][m + 1] + matrix[k][m - 1] - 4 * matrix[k][m]
    return result


def r_xx(matrix, s, t, n):
    return np.eye(n + 1) + s * t * t * lambda_xx(matrix, n)


def r_yy(matrix, s, t, n):
    return np.eye(n + 1) + s * t * t * lambda_yy(matrix, n)


def r(matrix, s, t, n):
    return matrix - s * t * t * (lambda_xx(matrix, n) + lambda_yy(matrix, n))
