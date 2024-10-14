# PK-MIQP
This is the code implementation of **Piecewise-linear Kernel Mixed Integer Quadratic Programming (PK-MIQP)**, an MIP-based minimizer with GP kernel approximation.

# Gurobi Requirement
PK-MIQP is built on Gurobi solver, which requires a license for downloading and use. Please refer to [this link](https://www.gurobi.com/features/academic-named-user-license/ "Academic License") for obtaining an academic license and [this link](https://www.gurobi.com/downloads/gurobi-software/, "Download") for downloading Gurobi v11.0.0. For non-academic users, please visit [here](https://www.gurobi.com/free-trial/ "Trial") to view instruction on getting a free trial. Note we evaluated performance of PK-MIQP using an academic license, model's functionalities might not be gauranteed when using Gurobi trial license.

# Get started
* Install required Python packages using [requirements.txt](requirements.txt)
```
pip install -r requirements.txt
```

*  [BayesOpt.py](BayesOpt.py) contains a BO implementation compiled with different minimizers for solving acquisition function LCB. Supported acquisition solvers include:
```
"PK-MIQP", "PK-MIQP-add", "L-BFGS-B", "Nelder-Mead", "COBYLA", "SLSQP", "trust-constr"
```
To minimize a black-box function, simply specify a black-box function, a method name and number of iterations for `BayesOpt` and run it.
*  As an example, run [BayesOpt.py](BayesOpt.py) directly will minimize the 1D `Bumpy` benchmark function using `PK-MIQP` and `L-BFGS-B` separately.
```
fun = Bumpy()
method_1 = "PK-MIQP"
x1, y1, log1 = BayesOpt(function=fun, method=method_1, n_iter=5, n_init_samples=10)

method_2 = "L-BFGS-B"
x2, y2, log2 = BayesOpt(function=fun, method=method_2, n_iter=5, n_init_samples=10)
```
* In [Functions](Functions) you can find more implemented [benchmarks functions](Functions/benchmarks.py) and an [SVM classifier hyperparameter tuning task as a black-box optimization function](Functions/SVM.py).
* [PK_MIQP.py](PK_MIQP.py) contains the implementation of `PK-MIQP` and `PK-MIQP-add` (PK-MIQP based on additive GP).
* [models.py](models.py) contains implementation of GP and additive GP.
* [utils.py](utils.py) contains the LCB function and Latin hypercube sampling function.
