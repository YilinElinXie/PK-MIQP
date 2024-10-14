import gpflow
import random
import numpy as np
import copy
import tensorflow_probability as tfp

class GP():
    '''Implementation of Gaussian process model
    Args:
        X (array): Sample points, shape (N, D)
        Y (array): Objective value of sample points, shape (N, 1)
        k (class): kernel class, default to Matern 3/2 kernel
        m (class): initial mean of GP model, default to None which is equivalent to 0
        var_range (tuple): bounds on kernel variance, default to (0.05, 20)
        ls_range (tuple): bounds on kernel lengthscales, default to (0.005, 2)
    '''

    def __init__(self, X, Y):

        self.X = X
        self.Y = Y
        self.k = gpflow.kernels.Matern32()
        self.m = None
        self.var_range = (0.05, 20.0)
        self.ls_range = (0.005, 2.0)

    def optimize(self, kernel):

        # build GP model using given kernel
        gpflow_model = gpflow.models.GPR(data=(self.X, self.Y), kernel=kernel,
                                         mean_function=self.m)
        opt = gpflow.optimizers.Scipy()
        # minimize negative log likelihood
        opt_logs = opt.minimize(gpflow_model.training_loss, gpflow_model.trainable_variables, options=dict(maxiter=100))

        return opt_logs.fun, kernel

    def train(self):

        # implement 10 multi-start softclip values
        var_lb, var_ub = self.var_range
        ls_lb, ls_ub = self.ls_range
        # Randomly select initial value of kernel variance and lengthscales within bounds
        var_ls_list = [(random.uniform(var_lb, var_ub), random.uniform(ls_lb, ls_ub)) for i in
                            range(10)]

        Loss = np.inf
        Kernel = None

        # iterate over different initial values, choose the one with the lowest loss
        for var_value, ls_value in var_ls_list:

            kernel = copy.deepcopy(self.k)

            # multi start softclip
            kernel.variance = gpflow.Parameter(
                var_value,
                transform=tfp.bijectors.SoftClip(
                    gpflow.utilities.to_default_float(var_lb),
                    gpflow.utilities.to_default_float(var_ub),
                ),
            )
            kernel.lengthscales = gpflow.Parameter(
                ls_value,
                transform=tfp.bijectors.SoftClip(
                    gpflow.utilities.to_default_float(ls_lb),
                    gpflow.utilities.to_default_float(ls_ub),
                ),
            )

            loss, kernel = self.optimize(kernel)

            if loss < Loss:
                Loss = loss
                Kernel = kernel

        self.kernel = Kernel



class Add_GP():
    '''Implementation of Gaussian process model
    Args:
        X (array): Sample points, shape (N, D)
        Y (array): Objective value of sample points, shape (N, 1)
        decomposition (list): a list consisting lists of dimensions, default to [[0], [1], ... , [D-1]]
        k (class): kernel class, default to Matern 3/2 kernel
        m (class): initial mean of GP model, default to None which is equivalent to 0
        var_range (tuple): bounds on kernel variance, default to (0.05, 20)
        ls_range (tuple): bounds on kernel lengthscales, default to (0.005, 2)
    '''

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.decomposition = [[i] for i in range(self.X.shape[1])]
        self.k = gpflow.kernels.Matern32()
        self.m = None
        self.var_range = (0.05, 20.0)
        self.ls_range = (0.005, 2.0)

    def optimize(self, kernel):

        kernel_list = []

        # Build additive kernel
        for i, group in enumerate(self.decomposition):
            base_kernel = copy.deepcopy(kernel)
            base_kernel.active_dims = group
            kernel_list.append(base_kernel)

        additive_kernel = np.sum(kernel_list)

        # Train GP model with additive kernel
        gpflow_model = gpflow.models.GPR(data=(self.X, self.Y), kernel=additive_kernel,
                                         mean_function=self.m)
        opt = gpflow.optimizers.Scipy()
        opt_logs = opt.minimize(gpflow_model.training_loss, gpflow_model.trainable_variables, options=dict(maxiter=100))

        return opt_logs.fun, gpflow_model.kernel.kernels


    def train(self):

        # implement 10 multi-start sofrclip values
        var_lb, var_ub = self.var_range
        ls_lb, ls_ub = self.ls_range
        # Randomly select initial value of kernel variance and lengthscales within bounds
        var_ls_list = [(random.uniform(var_lb, var_ub), random.uniform(ls_lb, ls_ub)) for i in
                       range(10)]

        Loss = np.inf
        Kernels = None

        # iterate over different initial values, choose the one with the lowest loss
        for var_value, ls_value in var_ls_list:

            kernel = copy.deepcopy(self.k)

            # multi start softclip
            var_lb, var_ub = self.var_range
            ls_lb, ls_ub = self.ls_range
            kernel.variance = gpflow.Parameter(
                var_value,
                transform=tfp.bijectors.SoftClip(
                    gpflow.utilities.to_default_float(var_lb),
                    gpflow.utilities.to_default_float(var_ub),
                ),
            )
            kernel.lengthscales = gpflow.Parameter(
                ls_value,
                transform=tfp.bijectors.SoftClip(
                    gpflow.utilities.to_default_float(ls_lb),
                    gpflow.utilities.to_default_float(ls_ub),
                ),
            )

            loss, kernels = self.optimize(kernel)

            if loss < Loss:
                Loss = loss
                Kernels = kernels

        self.kernels = Kernels
