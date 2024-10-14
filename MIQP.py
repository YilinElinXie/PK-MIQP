import copy
import gpflow
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum
import os
from scipy.optimize import minimize, Bounds
from models import GP
from utils import *


class PK_MIQP():

    '''Implementation of PK-MIQP minimizer
    Args:
        GPmodel (class): GP model
        beta (float): beta value of LCB
        Log (Boolean): MIQP model parameter, set True to show Gurobi solving process
        Warm_Start (Boolean): whether to initialize solving with sub-problem and random feasible solutions
        Time_Limit_SUB (int): time limit on solving time for sub-problem
        Time_Limit_FULL (int): time limit on solving time for full-problem
    '''

    def __init__(self,
                 GPmodel,
                 beta,
                 Log=False,
                 Warm_Start=True,
                 Time_Limit_SUB=600,
                 Time_Limit_FULL=1800,
                 ):

        self.GPmodel = GPmodel
        self.beta = beta
        self.N = self.GPmodel.X.shape[0]
        self.D = self.GPmodel.X.shape[1]
        self.Log = Log
        self.Warm_Start = Warm_Start
        self.Time_Limit_SUB = Time_Limit_SUB
        self.Time_Limit_FULL = Time_Limit_FULL


    def LCB(self, x):

        k = self.GPmodel.kernel
        X = self.GPmodel.X
        Y = self.GPmodel.Y
        lcb_fun = LCB(self.beta, k, X, Y)

        return lcb_fun(x)


    def piecewise_linearization(self):

        # compute some constant parameters needed in the following computation
        K_XX_list = []
        signal_variance = self.GPmodel.kernel.variance.numpy()
        self.lengthscales = self.GPmodel.kernel.lengthscales.numpy()
        self.K_xx = signal_variance


        # generate list of x and k(r) for addGenConstrPWL() to model the kernel function k(r)
        r_min = 0
        r_max = np.sqrt(self.D) / self.lengthscales
        r_1 = 0.486599
        r_2 = 0.711273
        r_3 = 2.12369

        if r_max <= r_1:
            self.R = np.linspace(0, r_max, 3)
        elif r_1 < r_max <= r_2:
            self.R = np.append(np.linspace(0, r_1, 2, endpoint=False),
                                  np.linspace(r_1, r_max, 2))
        elif r_2 < r_max <= r_3:
            self.R = np.append(np.append(np.linspace(0, r_1, 2, endpoint=False),
                                            np.linspace(r_1, r_2, 1, endpoint=False)),
                                  np.linspace(r_2, r_max, 3))
        else:
            self.R = np.append(
                np.append(np.append(np.linspace(0, r_1, 2, endpoint=False),
                                    np.linspace(r_1, r_2, 1, endpoint=False)),
                          np.linspace(r_2, r_3, 2, endpoint=False)),
                np.linspace(r_3, r_max, 3))

        self.k_R = self.GPmodel.kernel.K_r(self.R)

        for i in range(self.N):
            x = self.GPmodel.X[i]
            _, k = self.approx_kernel_value(x)
            K_XX_list.append(k)

        K_XX = np.array(K_XX_list).reshape(self.N, self.N)
        epsilon = -np.min([np.min(np.linalg.eigvals(K_XX)), 0]) + 1e-6

        K_XX = K_XX + epsilon * np.eye(self.N)

        self.K_XX_inv = np.linalg.pinv(K_XX)
        self.K_XX_inv_Y = np.dot(self.K_XX_inv, self.GPmodel.Y).flatten()


    def approx_kernel_value(self, x):

        r_xX = np.array([np.linalg.norm(x - self.GPmodel.X[i]) / self.lengthscales for i in range(self.N)])

        # piecewise function for calculating K_xX values
        def linear(x1, y1, x2, y2):
            m = (y1 - y2) / (x1 - x2)
            k = y1 - x1 * m
            return m, k

        condition_list = [((r_xX > self.R[i]) & (r_xX <= self.R[i + 1])) for i in
                          range(len(self.R) - 1)]
        condition_list[0] = ((r_xX >= self.R[0]) & (r_xX <= self.R[1]))
        conditions = np.transpose(np.array(condition_list).nonzero())
        mk_list = [linear(self.R[i], self.k_R[i], self.R[i + 1], self.k_R[i + 1]) for i in
                   range(len(self.R) - 1)]
        K_xX = np.zeros_like(r_xX)
        for index in conditions:
            K_xX[index[1]] = mk_list[index[0]][0] * r_xX[index[1]] + mk_list[index[0]][1]

        return r_xX, K_xX

    def build_MIP(self, indicator):

        if indicator == "SUB":

            model_name = "MIQP_SUB"

        elif indicator == "FULL":

            model_name = "MIQP_FULL"

        MIPmodel = gp.Model(model_name)

        x = MIPmodel.addMVar(self.D, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")
        x_list = x.tolist()

        m = MIPmodel.addMVar(1, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="m")


        self.K_xX = MIPmodel.addMVar(self.N, vtype=GRB.CONTINUOUS, name="K_xX")
        K_xX_list = self.K_xX.tolist()

        r_xX = MIPmodel.addMVar(self.N, vtype=GRB.CONTINUOUS, name="r_xX")
        r_xX_list = r_xX.tolist()

        if indicator == "FULL":
            std = MIPmodel.addMVar(1, vtype=GRB.CONTINUOUS, name="std")
            MIPmodel.setObjective(m - self.beta * std, GRB.MINIMIZE)
        elif indicator == "SUB":
            MIPmodel.setObjective(m, GRB.MINIMIZE)

        if not self.Log:
            MIPmodel.Params.LogToConsole = 0

        MIPmodel.Params.NonConvex = 2
        # Scale objective
        MIPmodel.Params.ObjScale = -0.5  # scale by square root of largest coefficient
        # Scale coefficients of matrix constraints
        MIPmodel.Params.ScaleFlag = 1  # equilibrium scaling

        # Add constraints:
        #        m - K_x,X*(K_XX)^(-1)*y = 0
        MIPmodel.addConstr(self.K_xX @ self.K_XX_inv_Y == m)
        if indicator == "FULL":
            MIPmodel.addConstr(self.K_xx - self.K_xX @ self.K_XX_inv @ self.K_xX >= std @ std)

        # Add constraints to model kernel:
        for i in range(self.N):
            # constraint on r
            MIPmodel.addConstr(
                quicksum((x_list[j] - self.GPmodel.X[i][j]) * (x_list[j] - self.GPmodel.X[i][j]) for j in
                         range(self.GPmodel.X.shape[1])) == r_xX_list[i] *
                r_xX_list[i] * (self.lengthscales ** 2))
            # PWL constraint on kernel
            MIPmodel.addGenConstrPWL(r_xX_list[i], K_xX_list[i], self.R, self.k_R)

        # Set some useful parameters to control the MIP solver
        MIPmodel.Params.TimeLimit = self.Time_Limit_SUB
        MIPmodel.Params.PoolSolutions = 10
        MIPmodel.Params.MIPGap = 0.05

        MIPmodel.update()

        return MIPmodel

    def update_MIP(self, FullModel, SubModel):

        # Build random solution pool
        random_sols = []
        xs = np.random.rand(500 * self.D, self.D)

        for x in xs:
            r_xX, K_xX = self.approx_kernel_value(x)
            posterior_variance = self.K_xx - K_xX @ self.K_XX_inv @ K_xX
            if posterior_variance >= 0:
                random_sols.append(x)

                if len(random_sols) >= 10:
                    break


        # Save random feasible solutions
        for idx, x in enumerate(random_sols):

            r_xX, K_xX = self.approx_kernel_value(x)
            m = K_xX @ self.K_XX_inv_Y
            var = self.K_xx - K_xX @ self.K_XX_inv @ K_xX
            std = np.sqrt(var)

            self.vars = np.real(np.concatenate((x, m, std, K_xX, r_xX), axis=None))

            var_names = [f"x[{i}]" for i in range(self.D)] + ["m[0]"] + ["std[0]"] + [f"K_xX[{i}]" for i in
                                                                                                   range(self.N)] + [
                                 f"r_xX[{i}]" for i in range(self.N)]
            lines = ["{} {}".format(v1, v2) for v1, v2 in zip(var_names, self.vars)]
            with open(f"sub_models/{SubModel.ModelName}_{SubModel.SolCount + idx}.sol", 'w') as f:
                f.write("\n".join(lines))

        # Warm start given FullModel
        FullModel.NumStart = SubModel.SolCount + len(random_sols)

        #    iterate over all MIP starts
        FullModel.update()
        for s in range(FullModel.NumStart):
            #    set StartNumber
            FullModel.params.StartNumber = s
            with open(f"sub_models/{SubModel.ModelName}_{s}.sol", 'r') as f:
                for line in f:
                    line.strip()
                    if len(line.split()) > 2:
                        (names, xn) = line.split()[:2]
                        # postprocessing xn
                        xn = xn[:14]
                    else:
                        (names, xn) = line.split()
                    for var in FullModel.getVars():
                        if var.varName == names:
                            var.Start = np.array(xn)

        FullModel.update()

        return FullModel


    def optimize_and_save(self, model):

        model.optimize()

        # Error handling
        if model.status == 3:

            print(
                "Infeasible sub model or unable to find solution for sub model.")

        else:

            # save solutions of the sub-model
            gv = model.getVars()
            names = model.getAttr('VarName', gv)
            for i in range(model.SolCount):
                model.params.SolutionNumber = i
                xn = model.getAttr('Xn', gv)
                lines = ["{} {}".format(v1, v2) for v1, v2 in zip(names, xn)]
                if not os.path.exists("sub_models/"):
                    os.makedirs("sub_models/")
                with open(f"sub_models/{model.ModelName}_{i}.sol", 'w') as f:
                    f.write("\n".join(lines))

    def pick_from_pool(self, model):

        best_x = None
        best_lcb = None
        distant_point = False

        gv = model.getVars()

        for i in range(model.SolCount):

            model.params.SolutionNumber = i
            xn = np.array(model.getAttr('Xn', gv)[:self.D])
            lcb = self.LCB(xn)
            distances = np.array([np.linalg.norm((xn - self.GPmodel.X[i])) for i in range(self.N)])
            also_distant_point = np.all(distances > 1e-4)

            if best_lcb is None or lcb < best_lcb:

                best_x = xn.reshape(-1, self.D)
                best_lcb = lcb

                if also_distant_point and not distant_point:
                    distant_point = True

        return best_x


    def solve(self, correction=True):

        self.piecewise_linearization()

        if self.Warm_Start:

            SUBmodel = self.build_MIP(indicator="SUB")
            self.optimize_and_save(SUBmodel)

            FULLmodel = self.build_MIP(indicator="FULL")
            self.update_MIP(FullModel=FULLmodel, SubModel=SUBmodel)

        else:
            FULLmodel = self.build_MIP(indicator="FULL")

        FULLmodel.optimize()

        best_x = self.pick_from_pool(FULLmodel)
        print(f"Best x picked from pool: {best_x}, lcb value: {self.LCB(best_x.flatten())}")

        if correction:
            # Correction by gradient solver
            result = minimize(self.LCB, x0=best_x.flatten(), method="L-BFGS-B",
                              bounds=Bounds([0] * self.D, [1] * self.D), options={"maxiter": 1000})
            x_next = np.array(result.x).reshape(1, -1)
            print(f"next x after correction: {x_next}, lcb value: {self.LCB(x_next.flatten())}")
        else:
            x_next = best_x

        return x_next

class PK_MIQPadd():
    '''Implementation of PK-MIQP minimizer equipped with Add-GP
    Args:
        AGPmodel (class): add-GP model
        beta (float): beta value of LCB
    '''

    def __init__(self,
                 AGPmodel,
                 beta):
        self.AGPmodel = AGPmodel
        self.beta = beta
        self.D = self.AGPmodel.X.shape[1]

    def solve(self):

        x_next = np.zeros((1, self.D))

        # Iterate over all dimension groups, use a PK-MIQP to solve each group
        for idx, dims in enumerate(self.AGPmodel.decomposition):

            kernel = copy.deepcopy(self.AGPmodel.kernels[idx])
            kernel.active_dims = None
            X_batch = self.AGPmodel.X[:, dims]
            Y_batch = self.AGPmodel.Y[:, -1]

            batch_GPmodel = GP(X_batch, Y_batch)
            batch_GPmodel.kernel = kernel

            MIP4batch = PK_MIQP(GPmodel=batch_GPmodel, beta=self.beta)
            x_batch_sol = MIP4batch.solve(correction=False)

            x_next[0][dims] = x_batch_sol.flatten()

        # Correction by gradient solver
        GPmodel = GP(self.AGPmodel.X, self.AGPmodel.Y)
        GPmodel.train()
        result = minimize(LCB(beta=self.beta, k=GPmodel.kernel, X=self.AGPmodel.X, Y=self.AGPmodel.Y), x0=x_next.flatten(), method="L-BFGS-B",
                          bounds=Bounds([0] * self.D, [1] * self.D), tol=1e-6,
                          options={"maxiter": 1000})
        return np.array(result.x).reshape(1, -1)
