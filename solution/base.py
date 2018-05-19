from utils import make_matrix
from solution.original import solve_layer, converge_solution, exact_solution


def solve_equation(n, eps=0.01):
    r = make_matrix(n)
    for k in range(n):
        solve_layer(r, n, k)

    r, diff = converge_solution(r, n)
    k = 0
    while diff > eps:
        k += 1
        r, diff = converge_solution(r, n)
    print(f'Сошлось за {k} итераций.')
    return r
