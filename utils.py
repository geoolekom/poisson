import numpy as np


def make_matrix(n):
    return np.empty([n + 1, n + 1], dtype=object)


def f(y, x):
    return - np.sin(np.pi * x) * np.cos(np.pi * y / 2) * 5 * np.pi * np.pi / 8
