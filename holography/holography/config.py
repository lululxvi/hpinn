import numpy as np


BOX = np.array([[-2, -2], [2, 3]])
DPML = 1

OMEGA = 2 * np.pi

SIGMA0 = -np.log(1e-20) / (4 * DPML ** 3 / 3)


def J(x):
    # Approximate the delta function
    y = x[:, 1:] + 1.5
    # hat function of width 2 * h
    # h = 0.5
    # return 1 / h * np.maximum(1 - np.abs(y / h), 0)
    # normal distribution of width ~2 * 2.5h
    h = 0.2
    return 1 / (h * np.pi ** 0.5) * np.exp(-((y / h) ** 2)) * (np.abs(y) < 0.5)
    # constant function of width 2 * h
    # h = 0.25
    # return 1 / (2 * h) * (np.abs(y) < h)


def target_square(X):
    f1 = np.heaviside((X[:, :1] + 0.5) * (0.5 - X[:, :1]), 0.5)
    f2 = np.heaviside((X[:, 1:] - 1) * (2 - X[:, 1:]), 0.5)
    return f1 * f2


def target_ring(X):
    d = np.linalg.norm(X - np.array([0, 1.5]), axis=1, keepdims=True)
    return np.heaviside((d - 0.9) * (1 - d), 0.5)


def target_face(X):
    c = dde.geometry.Disk([0, 1.5], 1) - dde.geometry.Disk([0, 1.5], 0.9)
    eye1 = dde.geometry.Disk([-0.5, 1.8], 0.1)
    eye2 = dde.geometry.Disk([0.5, 1.8], 0.1)
    mouth = dde.geometry.Disk([0, 1.3], 0.4) & dde.geometry.Rectangle(
        [-0.4, 0], [0.4, 1.3]
    )
    face = c | eye1 | eye2 | mouth
    return np.array([int(face.inside(x)) for x in X])[:, None]


target = target_square


def getEcomp(i, part):
    if part not in ["Re", "Im"]:
        raise ValueError(f"part={part}")
    # if i == 1:
    #     idx = 0 if part == "Re" else 1
    # elif i == 2:
    #     idx = 2 if part == "Re" else 3
    # elif i == 3:
    #     idx = 4 if part == "Re" else 5
    idx = 0 if part == "Re" else 1
    return idx


def getE(outputs, i, part):
    idx = getEcomp(i, part)
    return outputs[:, idx : idx + 1]
