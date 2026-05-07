import numpy as np

from loss_constants import compute_sigma2


def generate_data(n, d, xi):
    X = np.random.randn(n, d)
    X = xi * X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-8)

    true_theta = np.random.randn(d)
    y = X @ true_theta

    return X, y, true_theta


def loss(theta, X, y):
    r = X @ theta - y
    return 0.5 * np.mean(r ** 2)


def _squared_grad_i(theta, x, y):
    return (x @ theta - y) * x


def _prox_l1(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)


def _proj_box(x, kappa):
    return np.clip(x, -kappa, kappa)


def _sample_wishart(d, m):
    G = np.random.randn(d, m)
    return G @ G.T


def _clip(grad, zeta):
    norm = np.linalg.norm(grad)
    if norm <= zeta:
        return grad

    return grad / norm


def _run_stotos(X, y, grad_fn, kappa, omega, n, d, K, lam_l1):
    x = np.zeros(d)
    theta = np.zeros(d)

    for k in range(K):
        i = np.random.randint(n)

        theta = _prox_l1(x, omega * lam_l1)
        grad = n * grad_fn(theta, X[i], y[i])

        z = _proj_box(2 * theta - x - omega * grad, kappa)

        lambda_k = (k + 1) ** (-1 / 2 - 2 * 0.001)
        x = x - lambda_k * theta + lambda_k * z

    return theta


def solve_lop(X, y, reference_theta, eps, delta, kappa, n, d, xi, K, lam_l1, rho):
    L = d * xi ** 2
    zeta = (kappa * np.sqrt(d) * xi * np.sqrt(d) + max(y)) * xi * np.sqrt(d)

    Delta = 2 * L / eps
    sigma = zeta * np.sqrt(8 * np.log(2 / delta) + 4 * eps) / eps
    a = np.random.normal(0, sigma, size=d)

    omega = 1 / (n * L + Delta)

    def grad(theta, x, yi):
        return _squared_grad_i(theta, x, yi) + Delta * theta / n + a / n

    return _run_stotos(X, y, grad, kappa, omega, n, d, K, lam_l1)


def solve_lop_clipped(X, y, reference_theta, eps, delta, kappa, n, d, xi, K, lam_l1, rho):
    L = d * xi ** 2
    zeta = 10000  # (kappa * np.sqrt(d) * xi + 1) * xi

    Delta = 2 * L / eps
    sigma = zeta * np.sqrt(8 * np.log(2 / delta) + 4 * eps) / eps
    a = np.random.normal(0, sigma, size=d)

    omega = 1 / (n * L + Delta)

    def grad(theta, x, yi):
        return _clip(_squared_grad_i(theta, x, yi), zeta) + Delta * theta / n + a / n

    return _run_stotos(X, y, grad, kappa, omega, n, d, K, lam_l1)


def solve_qop(X, y, reference_theta, eps, delta, kappa, n, d, xi, K, lam_l1, rho):
    L = d * xi ** 2

    m = 2 * d
    sigma2 = compute_sigma2(L, eps, delta / 2, delta / 2, d, m, rho)

    W = _sample_wishart(d, m)
    lambda_max = np.linalg.norm(W, 2)

    omega = 1 / (n * L + sigma2 * lambda_max)

    def grad(theta, x, yi):
        return _squared_grad_i(theta, x, yi) + sigma2 * W @ (theta - reference_theta) / n

    return _run_stotos(X, y, grad, kappa, omega, n, d, K, lam_l1)
