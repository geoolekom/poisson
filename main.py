from plotting import function_plot, solution_plot, show_plot, err_plot
from solution.base import solve_equation
from solution.original import exact_solution


def main():
    n = input('Введите N:\n')
    if n.isdigit():
        n = int(n)
        u = solve_equation(n)
        function_plot(n + 1, exact_solution)
        solution_plot(n + 1, u)
        err_plot(n + 1, exact_solution, u)
        show_plot()
    else:
        print('Введите целое число.')


if __name__ == '__main__':
    main()
