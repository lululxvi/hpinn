import numpy as np
from deepxde.backend import tf

from .config import BOX, DPML, OMEGA


# def interface12(x, _):
#     return np.isclose(x[1], -1)


# def interface23(x, _):
#     return np.isclose(x[1], 0)


# def interface12_ReE(inputs, outputs, X):
#     return getE(outputs, 1, "Re") - getE(outputs, 2, "Re")


# def interface12_ImE(inputs, outputs, X):
#     return getE(outputs, 1, "Im") - getE(outputs, 2, "Im")


# def interface12_dReE(inputs, outputs, X):
#     return dde.grad.jacobian(
#         outputs, inputs, i=getEcomp(1, "Re"), j=1
#     ) - dde.grad.jacobian(outputs, inputs, i=getEcomp(2, "Re"), j=1)


# def interface12_dImE(inputs, outputs, X):
#     return dde.grad.jacobian(
#         outputs, inputs, i=getEcomp(1, "Im"), j=1
#     ) - dde.grad.jacobian(outputs, inputs, i=getEcomp(2, "Im"), j=1)


# def interface23_ReE(inputs, outputs, X):
#     return getE(outputs, 2, "Re") - getE(outputs, 3, "Re")


# def interface23_ImE(inputs, outputs, X):
#     return getE(outputs, 2, "Im") - getE(outputs, 3, "Im")


# def interface23_dReE(inputs, outputs, X):
#     return dde.grad.jacobian(
#         outputs, inputs, i=getEcomp(2, "Re"), j=1
#     ) - dde.grad.jacobian(outputs, inputs, i=getEcomp(3, "Re"), j=1)


# def interface23_dImE(inputs, outputs, X):
#     return dde.grad.jacobian(
#         outputs, inputs, i=getEcomp(2, "Im"), j=1
#     ) - dde.grad.jacobian(outputs, inputs, i=getEcomp(3, "Im"), j=1)


# def boundary1_leftright(x, on_boundary):
#     return x[1] <= -1 and np.isclose(abs(x[0]), 6)


# def boundary1_bottom(x, on_boundary):
#     return np.isclose(x[1], -3)


# def boundary2(x, on_boundary):
#     return on_boundary and -1 <= x[1] <= 0


# def boundary3_leftright(x, on_boundary):
#     return x[1] >= 0 and np.isclose(abs(x[0]), 6)


# def boundary3_top(x, on_boundary):
#     return np.isclose(x[1], 6)


def feature_transform(inputs):
    # Periodic BC in x
    P = BOX[1][0] - BOX[0][0] + 2 * DPML
    w = 2 * np.pi / P
    x, y = w * inputs[:, :1], inputs[:, 1:]
    return tf.concat(
        (
            tf.math.cos(x),
            tf.math.sin(x),
            tf.math.cos(2 * x),
            tf.math.sin(2 * x),
            tf.math.cos(3 * x),
            tf.math.sin(3 * x),
            tf.math.cos(4 * x),
            tf.math.sin(4 * x),
            tf.math.cos(5 * x),
            tf.math.sin(5 * x),
            tf.math.cos(6 * x),
            tf.math.sin(6 * x),
            # tf.math.cos(7 * x),
            # tf.math.sin(7 * x),
            # tf.math.cos(8 * x),
            # tf.math.sin(8 * x),
            # tf.math.cos(9 * x),
            # tf.math.sin(9 * x),
            # tf.math.cos(10 * x),
            # tf.math.sin(10 * x),
            y,
            tf.math.cos(OMEGA * y),
            tf.math.sin(OMEGA * y),
        ),
        axis=1,
    )


def output_transform(inputs, outputs):
    x, y = inputs[:, :1], inputs[:, 1:]

    # 1 <= eps <= 12
    eps = 1 + 11 * tf.math.sigmoid(outputs[:, -1:])

    # Zero Dirichlet BC
    a, b = BOX[0][1] - DPML, BOX[1][1] + DPML
    E = (1 - tf.math.exp(a - y)) * (1 - tf.math.exp(y - b)) * outputs[:, :2]

    # Zero Dirichlet and Neumann BC
    # a, b = BOX[0][1] - DPML, BOX[1][1] + DPML
    # E = 0.01 * (a - y) ** 2 * (y - b) ** 2 * outputs[:, :2]

    # return E
    return tf.concat((E, eps), axis=1)


# def output_transform(inputs, outputs):
#     a, b = -3, 6
#     y = inputs[:, 1:]
#     E1 = (1 - tf.math.exp(a - y)) * outputs[:, :2]
#     E2 = outputs[:, 2:4]
#     E3 = (1 - tf.math.exp(y - b)) * outputs[:, 4:6]
#     eps = 1 + 11 * tf.math.sigmoid(outputs[:, -1:])
#     # return tf.concat([outputs[:, :-1], eps], 1)
#     return tf.concat((E1, E2, E3, eps), 1)


# def output_transform(inputs, outputs):
#     x, y = inputs[:, :1], inputs[:, 1:]

#     # 1 <= eps <= 12
#     eps = 1 + 11 * tf.math.sigmoid(outputs[:, -1:])

#     # Continuity of E and dE/dy
#     P = 12
#     c, d = -1, 0
#     w = 2 * np.pi / P
#     e = tf.concat(
#         (
#             tf.math.cos(w * x),
#             tf.math.sin(w * x),
#             tf.math.cos(2 * w * x),
#             tf.math.sin(2 * w * x),
#         ),
#         axis=1,
#     )
#     units = 16
#     ReN1 = tf.keras.layers.Dense(units, activation="tanh")(e)
#     ReN1 = tf.keras.layers.Dense(1)(ReN1) * 0.1
#     ReN2 = tf.keras.layers.Dense(units, activation="tanh")(e)
#     ReN2 = tf.keras.layers.Dense(1)(ReN2) * 0.1
#     ImN1 = tf.keras.layers.Dense(units, activation="tanh")(e)
#     ImN1 = tf.keras.layers.Dense(1)(ImN1) * 0.1
#     ImN2 = tf.keras.layers.Dense(units, activation="tanh")(e)
#     ImN2 = tf.keras.layers.Dense(1)(ImN2) * 0.1
#     y1 = y - c
#     y1 = y1 * tf.math.abs(y1)
#     y2 = y - d
#     y2 = y2 * tf.math.abs(y2)
#     ReE = outputs[:, :1] + ReN1 * y1 + ReN2 * y2
#     ImE = outputs[:, 1:2] + ImN1 * y1 + ImN2 * y2
#     E = tf.concat((ReE, ImE), axis=1)

#     # Zero Dirichlet BC
#     a, b = -3, 6
#     E = (1 - tf.math.exp(a - y)) * (1 - tf.math.exp(y - b)) * E
#     return tf.concat((E, eps), axis=1)
