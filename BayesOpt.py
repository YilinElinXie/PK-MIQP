from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
from models import *
from PK_MIQP import *
from utils import *
from Functions.benchmarks import *


def BayesOpt(function, method, n_iter, n_init_samples=10):
    '''
    BO algorithm to minimize a test function.
    Args:
        function (class): objective function class, select from Functions repo
        method (str): name of minimizer,
                    select from "PK-MIQP", "PK-MIQP-add", "L-BFGS-B", "Nelder-Mead", "COBYLA", "SLSQP", "trust-constr"
        n_iter (int): number of iterations
        n_init_samples (int): number of initial sample points

    Returns:
        argmin_x (array): point that minimize the objective function, shape (D,)
        min_y (float): minimum value of objective found
        iteration-wise_sample_points (array): point suggested by the solver at each iteration, shape (n_iter, D+1)
    '''
    # Generate initial set of sample points for GP
    data = LHS(function, n_init_samples)
    D = function.D
    X, Y = data[:, :-1].reshape(-1, D), data[:, -1].reshape(-1, 1)

    # Initialize x_opt and y_opt
    idx_opt = np.argmin(Y)
    y_opt = np.min(Y)
    log = []

    for i in range(n_iter):

        # scale samples
        bound = np.append(function.domain, np.array([[Y.min()], [Y.max()]]), axis=1)
        scaler = MinMaxScaler()
        scaler.fit(bound)
        scaled_data = scaler.transform(data)
        X_scaled = scaled_data[:, :-1].reshape(-1, D)
        Y_scaled = scaled_data[:, -1].reshape(-1, 1)

        if method == "PK-MIQP-add":
            # use add-GP model
            model = Add_GP(X_scaled, Y_scaled)
        else:
            model = GP(X_scaled, Y_scaled)

        # GP training
        model.train()

        # LCB function's beta
        beta = np.sqrt(0.2 * D * np.log(2 * (i+1)))

        if method in ["L-BFGS-B", "Nelder-Mead", "COBYLA", "SLSQP", "trust-constr"]:
            # starting point
            x0 = np.zeros(D,)
            obj = LCB(beta=beta, k=model.kernel, X=X_scaled, Y=Y_scaled)
            result = minimize(obj, x0, method=method, bounds=Bounds([0]*D, [1]*D), tol=1e-6,
                              options={"maxiter": 1000})
            x_next = np.array(result.x).reshape(1, -1)

        elif method == "PK-MIQP":

            solver = PK_MIQP(model, beta)
            x_next = solver.solve()

        elif method == "PK-MIQP-add":

            solver = PK_MIQPadd(model, beta)
            x_next = solver.solve()

        else:
            print(f"Method {method} not implemented.")

        # Add unscaled new sample to dataset
        dummy_y = np.array(0).reshape(-1, 1)
        x_next_unscaled = scaler.inverse_transform(np.append(x_next, dummy_y).reshape(1, -1))[:, :-1]
        y_next_unscaled = np.array(function.obj(x_next_unscaled.flatten())).reshape(-1, 1)
        data = np.vstack([data, np.append(x_next_unscaled, y_next_unscaled)])

        if y_next_unscaled < y_opt:
            y_opt = y_next_unscaled
            idx_opt = len(data) - 1  # since always add new points to last row

        # Log of iteration sample point
        log.append(data[-1].tolist())

        print(f"Iteration {i+1}/{n_iter}, point sampled: {data[-1]}")

    return data[idx_opt][:-1], y_opt, np.array(log)


if __name__ == "__main__":

    fun = Bumpy()
    method_1 = "PK-MIQP"
    x1, y1, log1 = BayesOpt(function=fun, method=method_1, n_iter=5, n_init_samples=10)
    print(f"PK-MIQP results: {x1, y1, log1}")

    method_2 = "L-BFGS-B"
    x2, y2, log2 = BayesOpt(function=fun, method=method_2, n_iter=5, n_init_samples=10)
    print(f"L-BFGS-B results: {x2, y2, log2}")


