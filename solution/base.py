import numpy as np

from utils import make_matrix, f
from solution.original import solve_layer, exact_solution
from solution.hyperbole import converge_solution


def first_approach(n):
    r = make_matrix(n)
    for k in range(n + 1):
        for m in range(n + 1):
            r[k][m] = exact_solution(m / n, k / n) * (1 + 0.1 * np.sin(m))
    for k in range(0, n + 1):
        r[k][0] = 0.0
        r[k][n] = 0.0
        r[0][k] = np.sqrt(np.sin(np.pi * k / n))
        r[n][k] = 0.0
    return r


def solve_equation(n, eps=0.001):
    r = make_matrix(n)
    for k in range(n):
        solve_layer(r, n, k)

    prev = np.array(r)
    r, diff, prev = converge_solution(r, n, prev)
    diff0 = diff
    k = 1
    while diff > diff0 * eps:
        r, diff, prev = converge_solution(r, n, prev)
        if k % 10 == 0 and k != 0:
            error = round(diff * 100 / diff0, ndigits=2)
            print(f'Прошло {k} итераций, погрешность - {error}%')
        k += 1

    print(f'Сошлось за {k} итераций.')
    return r
