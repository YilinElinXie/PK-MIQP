'''
Benchmarks function implementation.
All benchmarks functions are implemented as a class. Each class includes dimensionality (self.D), problem's search domain (self.domain) and global optimal value (self.opt_y). Objective function can be accessed by self.obj(x). 
'''
import numpy as np

class Gramacy_Lee():

    def __init__(self):

        self.name = "Gramacy_Lee"
        self.lb = [0.5]
        self.ub = [2.5]
        self.domain = np.array([[0.5],
                       [2.5]])
        self.D = 1
        self.opt_y = -0.8690111349895

    def obj(self, x):

        return np.sin(10 * np.pi * x[0]) / (2 * x[0]) + (x[0] - 1) ** 4


class Forrester():

    def __init__(self):
        self.name = "Forrester"
        self.lb = [0]
        self.ub = [1]
        self.domain = np.array([[0],
                       [1]])
        self.D = 1
        self.opt_y = -6.02074

    def obj(self, x):
        return (6 * x[0] - 2)**2 * np.sin(12 * x[0] - 4)


class Bumpy():
    # periodic with multiple local and global minima

    def __init__(self):
        self.name = "Bumpy"
        self.lb = [-10]
        self.ub = [10]
        self.domain = np.array([[-10],
                       [10]])
        self.D = 1
        self.opt_y = -16.532194721073317

    def obj(self, x):

        return - np.sum([i * np.sin((i + 1)*x[0] + i) for i in range(1, 7)])


class Multimodal():
    # multiple local minima and a unique global minimum

    def __init__(self):
        self.name = "Multimodal"
        self.lb = [-2.7]
        self.ub = [7.5]
        self.domain = np.array([[-2.7],
                       [7.5]])
        self.D = 1
        self.opt_y = -1.899599349151611

    def obj(self, x):

        return np.sin(x[0]) + np.sin(10 / 3 * x[0])


class Ackley():

    def __init__(self):
        self.name = "Ackley"
        self.lb = [-32, -32]
        self.ub = [16, 16]
        self.domain = np.array([[-32, -32],
                       [16, 16]])
        self.D = 2
        self.opt_y = 0

    def obj(self, x):

        a = 20
        b = 0.2
        c = 2 * np.pi
        d = 2
        return (-a * np.exp(-b * np.sqrt(1 / d * np.sum([x[i] ** 2 for i in range(d)])))
                - np.exp(1 / d * np.sum([np.cos(c * x[i]) for i in range(d)])) + a + np.exp(1))


class Branin():

    def __init__(self):
        self.name = "Branin"
        self.lb = [-5, 0]
        self.ub = [10, 15]
        self.domain = np.array([[-5, 0],
                       [10, 15]])
        self.D = 2
        self.opt_y = 0.397887

    def obj(self, x):

        a = 1
        b = 5.1 / (4 * np.pi ** 2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)

        return a * (x[1] - b * (x[0] ** 2) + c * x[0] - r) ** 2 + s * (1 - t) * np.cos(x[0]) + s


class Giunta():

    def __init__(self):
        self.name = "Giunta"
        self.lb = [-1, -1]
        self.ub = [1, 1]
        self.domain = np.array([[-1, -1],
                       [1, 1]])
        self.D = 2
        self.opt_y = 0.06447042053690566

    def obj(self, x):

        return 0.6 + np.sum([-np.sin(1 - 16 / 15 * x[i]) + (np.sin(1 - 16 / 15 * x[i])) ** 2
                             - 1 / 50 * np.sin(4 * (1 - 16 / 15 * x[i])) for i in range(2)])



class Rosenbrock():

    def __init__(self):
        self.name = "Rosenbrock"
        self.lb = [-2, -1]
        self.ub = [2, 3]
        self.domain = np.array([[-2, -1],
                       [2, 3]])
        self.D = 2
        self.opt_y = 0

    def obj(self, x):

        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


class Hartmann():

    def __init__(self):
        self.name = "Hartmann"
        self.lb = [0, 0, 0]
        self.ub = [1, 1, 1]
        self.domain = np.array([[0, 0, 0],
                       [1, 1, 1]])
        self.D = 3
        self.opt_y = -3.86278

    def obj(self, x):

        alpha = np.array([1, 1.2, 3, 3.2])
        A = np.array([[3, 10, 30],
                      [0.1, 10, 35],
                      [3, 10, 30],
                      [0.1, 10, 35]])
        P = 1e-4 * np.array([[3689, 1170, 2673],
                             [4699, 4387, 7470],
                             [1091, 8732, 5547],
                             [381, 5743, 8828]])
        f = 0

        for i in range(4):

            sub = 0

            for j in range(3):
                sub += A[i, j] * (x[j] - P[i, j]) ** 2

            f += alpha[i] * np.exp(-sub)

        return -f


class Michalewicz():

    def __init__(self):
        self.name = "Michalewicz"
        self.lb = [0, 0, 0, 0, 0]
        self.ub = [np.pi, np.pi, np.pi, np.pi, np.pi]
        self.domain = np.array([[0, 0, 0, 0, 0],
                       [np.pi, np.pi, np.pi, np.pi, np.pi]])
        self.D = 5
        self.opt_y = -4.687658

    def obj(self, x):

        return - np.sum([np.sin(x[i]) * (np.sin((i + 1) * (x[i] ** 2) / np.pi)) ** 20 for i in range(5)])
