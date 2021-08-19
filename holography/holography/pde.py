import deepxde as dde
import numpy as np

from .config import BOX, J, OMEGA, SIGMA0


def PML(X):
    def sigma(x, a, b):
        """sigma(x) = 0 if a < x < b, else grows cubically from zero.
        """

        def _sigma(d):
            return SIGMA0 * d ** 2 * np.heaviside(d, 0)

        return _sigma(a - x) + _sigma(x - b)

    def dsigma(x, a, b):
        def _sigma(d):
            return 2 * SIGMA0 * d * np.heaviside(d, 0)

        return -_sigma(a - x) + _sigma(x - b)

    sigma_x = sigma(X[:, :1], BOX[0][0], BOX[1][0])
    AB1 = 1 / (1 + 1j / OMEGA * sigma_x) ** 2
    A1, B1 = AB1.real, AB1.imag

    dsigma_x = dsigma(X[:, :1], BOX[0][0], BOX[1][0])
    AB2 = -1j / OMEGA * dsigma_x * AB1 / (1 + 1j / OMEGA * sigma_x)
    A2, B2 = AB2.real, AB2.imag

    sigma_y = sigma(X[:, 1:], BOX[0][1], BOX[1][1])
    AB3 = 1 / (1 + 1j / OMEGA * sigma_y) ** 2
    A3, B3 = AB3.real, AB3.imag

    dsigma_y = dsigma(X[:, 1:], BOX[0][1], BOX[1][1])
    AB4 = -1j / OMEGA * dsigma_y * AB3 / (1 + 1j / OMEGA * sigma_y)
    A4, B4 = AB4.real, AB4.imag
    return A1, B1, A2, B2, A3, B3, A4, B4


def pde(inputs, outputs, X, ReE, ImE, eps):
    A1, B1, A2, B2, A3, B3, A4, B4 = PML(X)

    dReE_x = dde.grad.jacobian(outputs, inputs, i=ReE, j=0)
    dReE_y = dde.grad.jacobian(outputs, inputs, i=ReE, j=1)
    dReE_xx = dde.grad.hessian(outputs, inputs, component=ReE, i=0, j=0)
    dReE_yy = dde.grad.hessian(outputs, inputs, component=ReE, i=1, j=1)
    dImE_x = dde.grad.jacobian(outputs, inputs, i=ImE, j=0)
    dImE_y = dde.grad.jacobian(outputs, inputs, i=ImE, j=1)
    dImE_xx = dde.grad.hessian(outputs, inputs, component=ImE, i=0, j=0)
    dImE_yy = dde.grad.hessian(outputs, inputs, component=ImE, i=1, j=1)

    ReE = outputs[:, ReE : ReE + 1]
    ImE = outputs[:, ImE : ImE + 1]

    loss_Re = (
        (A1 * dReE_xx + A2 * dReE_x + A3 * dReE_yy + A4 * dReE_y) / OMEGA
        - (B1 * dImE_xx + B2 * dImE_x + B3 * dImE_yy + B4 * dImE_y) / OMEGA
        + eps * OMEGA * ReE
    )
    loss_Im = (
        (A1 * dImE_xx + A2 * dImE_x + A3 * dImE_yy + A4 * dImE_y) / OMEGA
        + (B1 * dReE_xx + B2 * dReE_x + B3 * dReE_yy + B4 * dReE_y) / OMEGA
        + eps * OMEGA * ImE
        + J(X)
    )
    # return loss_Re, loss_Im
    return loss_Re, loss_Im, loss_Re, loss_Im  # augmented_Lagrangian


def pde_domain(inputs, outputs, X):
    condition = np.logical_and(X[:, 1:] < 0, X[:, 1:] > -1).astype(np.float32)
    eps = outputs[:, -1:] * condition + 1 - condition
    # eps = 1
    return pde(inputs, outputs, X, 0, 1, eps)


# def pde_bc(outputs, inputs, ReE, ImE, X, eps=1):
#     loss_Re, loss_Im = pde(inputs, outputs, X, ReE, ImE, eps)
#     return tf.concat([loss_Re, loss_Im], axis=1)


# def pde_E1_ReIm(inputs, outputs, X):
#     return pde_bc(outputs, inputs, getEcomp(1, "Re"), getEcomp(1, "Im"), X)


# def pde_E2_ReIm(inputs, outputs, X):
#     return pde_bc(
#         outputs, inputs, getEcomp(2, "Re"), getEcomp(2, "Im"), X, eps=outputs[:, -1:]
#     )


# def pde_E3_ReIm(inputs, outputs, X):
#     return pde_bc(outputs, inputs, getEcomp(3, "Re"), getEcomp(3, "Im"), X)
