import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=plt.figaspect(0.5))
func_ax: Axes3D = fig.add_subplot(121, projection='3d', )
func_ax.set_title('Точное и численное решение')
func_ax.set_xlim(0, 1)
func_ax.set_xlabel('X')
func_ax.set_ylim(0, 1)
func_ax.set_ylabel('Y')

err_ax: Axes3D = fig.add_subplot(122, projection='3d', )
err_ax.set_title('Карта ошибок')
err_ax.set_xlim(0, 1)
err_ax.set_xlabel('X')
err_ax.set_ylim(0, 1)
err_ax.set_ylabel('Y')


def function_plot(n, fn):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    x, y = np.meshgrid(x, y)
    z = fn(x, y)
    func_ax.plot_surface(x, y, z)


def solution_plot(n, z):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    x, y = np.meshgrid(x, y)
    func_ax.plot_wireframe(x, y, z, color=[0, 0.5, 0])


def err_plot(n, fn, approximate):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    x, y = np.meshgrid(x, y)
    z = approximate - fn(x, y)
    err_ax.plot_surface(x, y, z)


def show_plot():
    plt.show()
