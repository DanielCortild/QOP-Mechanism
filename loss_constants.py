import numpy as np

from scipy.special import gammainc, gammaincinv, gamma

inc_gamma = lambda x, a: gamma(a) * gammainc(a, x)
inv_inc_gamma = lambda y, a: gammaincinv(a, y / gamma(a))

def compute_Dm(d, m):
    return (d * gamma(3 / 2) * gamma((m + 1) / 2)) / (
            gamma(d / 2 + 1)
            * gamma((m - d + 1) / 2)
            * gamma((m - d + 2) / 2)
    )


def compute_alpha(del3, d, m):
    Dm = compute_Dm(d, m)
    return 2 * inv_inc_gamma(del3 / Dm, (m - d + 1) / 2)


def compute_alpha1(del1, del3, d, m):
    alpha = compute_alpha(del3, d, m)
    Dm = compute_Dm(d, m)
    shape = (m - d + 1) / 2
    return inv_inc_gamma(del1 * (1 - del3) / Dm + inc_gamma(alpha, shape), shape) - alpha


def compute_f2(del3, d, m):
    alpha = compute_alpha(del3, d, m)
    return 2 * ((m - d + 1) / (2 * alpha) + 1 / 2)


def compute_sigma2(L, eps1, del1, del3, d, m, rho):
    f2 = compute_f2(del3, d, m)
    alpha = compute_alpha(del3, d, m)
    alpha1 = compute_alpha1(del1, del3, d, m)
    return max(
        2 * L / eps1 * (f2 + 2 * (2 * rho + 2) / alpha),
        2 * L / alpha1,
    )


def compute_beta(del3, del4, d, m):
    return (
            np.sqrt(2 * np.log(2 / (del4 * (1 - del3)))) + np.sqrt(m) + np.sqrt(d)
    ) ** 2


def compute_sigma_tilde2(L, tau, eta, eps1, eps2, del1, del2, del3, del4, d, m, rho):
    if tau == 0:
        return 0

    alpha = compute_alpha(del3, d, m)
    sigma2 = compute_sigma2(L, eps1, del1, del3, d, m, rho)
    beta = compute_beta(del3, del4, d, m)
    return (
            (np.sqrt(2 * tau / (alpha * sigma2)) + beta * eta / alpha)
            * 2
            / eps2
            * np.sqrt(2 * np.log(1.25 / del2))
    ) ** 2

def compute_compute_loss(L, tau, n, d, D, rho, G, eta):
    def compute_loss(eps1, eps2, del1, del2, del3, del4, m):
        sigma_tilde2 = compute_sigma_tilde2(L, tau, eta, eps1, eps2, del1, del2, del3, del4, d, m, rho)
        sigma2 = compute_sigma2(L, eps1, del1, del3, d, m, rho)

        return (
                n * d * L * sigma_tilde2 / 2
                    + G * np.sqrt(d) * np.sqrt(sigma_tilde2)
                    + tau
                    + sigma2 * m * (D ** 2 + eta ** 2)
                ) if eta != 0 else sigma2 / 2 * m * D ** 2

    return compute_loss
