import numpy as np
from scipy.stats import qmc

def LHS(function, n_samples):
    '''
    Use Latin hypercube sampling to generate data
    '''
    D, lb, ub = function.D, function.lb, function.ub
    sampler = qmc.LatinHypercube(d=D)
    sample = sampler.random(n=n_samples)
    X = qmc.scale(sample, lb, ub)
    Y = np.array([
        function.obj(x) for x in X
    ]).reshape(-1, 1)
    data = np.append(X, Y, axis=1)

    return data


def LCB(beta, k, X, Y):
    '''
    Construct LCB function using given kernel information
    '''

    # Manually calculate mean and std
    epsilon = 1e-6
    K_XX = k(X, X) + epsilon * np.eye(len(X))
    K_XX_inv = np.linalg.pinv(K_XX)
    K_XX_inv_Y= np.dot(K_XX_inv, Y).flatten()
    K_xx = k.variance.numpy()
    lengthscales = k.lengthscales.numpy()


    def lcb(x):
        r_xX = np.array([np.linalg.norm((x - X[i]), ord=2) /
                                 lengthscales for i in range(len(X))])
        K_xX = k.K_r(r_xX).numpy()
        mean = K_xX @ K_XX_inv_Y
        variance = K_xx - K_xX @ K_XX_inv @ K_xX
        return mean - beta * np.sqrt(variance)

    return lcb