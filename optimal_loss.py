import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, softmax

from loss_constants import compute_compute_loss

L = 1  # Smoothness Constant of Loss Functions
n = 10  # Number of Datapoints
d = 12  # Ambient Dimension (must be larger than n)
D = 1  # Distance of Unperturbed Solution to Common Minimizer (Squared)
rho = 1  # Maximal Rank of Hessian of Loss Function
G = 1  # Bound of Subgradients of Regularizer

EXACT_TAU = 0
EXACT_ETA = 0
INEXACT_TAU = 0.001
INEXACT_ETA = 0.001


def compute_optimal_loss(eps, delta, exact=True):
    if eps <= 0:
        raise ValueError("eps must be positive.")
    if delta <= 0:
        raise ValueError("delta must be positive.")

    if exact:
        return _compute_exact_branch(eps, delta)

    return _compute_inexact_branch(eps, delta)


def _compute_exact_branch(eps, delta):
    compute_loss = compute_compute_loss(
        L, EXACT_TAU, n, d, D, rho, G, EXACT_ETA
    )

    def unpack(z):
        w = softmax(z[0:2])
        delta1, delta3 = delta * w
        m = z[2] ** 2 + d + 1

        return delta1, delta3, m

    def objective(z):
        delta1, delta3, m = unpack(z)
        val = compute_loss(eps, 0, delta1, 0, delta3, 0, m)

        if not np.isfinite(val):
            return 1e100

        return val

    x0 = np.zeros(3)
    x0[2] = 0.5

    res = minimize(
        objective,
        x0=np.asarray(x0, dtype=float),
        method="Nelder-Mead",
        options={"maxiter": 2000, "disp": False},
    )

    return res.fun


def _compute_inexact_branch(eps, delta):
    compute_loss = compute_compute_loss(
        L, INEXACT_TAU, n, d, D, rho, G, INEXACT_ETA
    )

    def unpack(z):
        frac_eps1 = expit(z[0])
        eps1 = eps * frac_eps1
        eps2 = eps - eps1

        w = softmax(z[1:5])
        delta1, delta2, delta3, delta4 = delta * w
        m = max(z[5], d)

        return eps1, eps2, delta1, delta2, delta3, delta4, m

    def objective(z):
        eps1, eps2, delta1, delta2, delta3, delta4, m = unpack(z)
        val = compute_loss(eps1, eps2, delta1, delta2, delta3, delta4, m)

        if not np.isfinite(val):
            return 1e100

        return val

    x0 = np.zeros(6)
    x0[5] = 0.5

    res = minimize(
        objective,
        x0=np.asarray(x0, dtype=float),
        method="Nelder-Mead",
        options={"maxiter": 20000, "disp": False},
    )

    return res.fun
