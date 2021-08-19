import deepxde as dde
import numpy as np
from deepxde.backend import tf

# tf.random.set_random_seed(1234)

GAMMA = 0.9


def save_solution(geom, model, filename):
    x = geom.uniform_points(40000)
    y_pred = model.predict(x)
    print("Saving u and p ...\n")
    np.savetxt(filename + "_fine.dat", np.hstack((x, y_pred, alpha(y_pred[:, -1:]))))

    x = geom.uniform_points(256)
    y_pred = model.predict(x)
    print("Saving u and p ...\n")
    np.savetxt(filename + "_coarse.dat", np.hstack((x, y_pred, alpha(y_pred[:, -1:]))))


def alpha(rho):
    alpha_max, alpha_min = 2.5 * 10 ** 4, 0  # 2.5 / 10 ** 4
    q = 0.1
    return alpha_max + (alpha_min - alpha_max) * rho * (1 + q) / (rho + q)


def pde(inputs, outputs):
    du_x = dde.grad.jacobian(outputs, inputs, i=0, j=0)
    dv_y = dde.grad.jacobian(outputs, inputs, i=1, j=1)
    du_xx = dde.grad.hessian(outputs, inputs, component=0, i=0, j=0)
    du_yy = dde.grad.hessian(outputs, inputs, component=0, i=1, j=1)
    dv_xx = dde.grad.hessian(outputs, inputs, component=1, i=0, j=0)
    dv_yy = dde.grad.hessian(outputs, inputs, component=1, i=1, j=1)
    dp_x = dde.grad.jacobian(outputs, inputs, i=2, j=0)
    dp_y = dde.grad.jacobian(outputs, inputs, i=2, j=1)
    f = alpha(outputs[:, 3:]) * outputs[:, :2]
    fx, fy = f[:, :1], f[:, 1:]
    loss1 = (-(du_xx + du_yy) + dp_x - fx) * 0.01
    loss2 = (-(dv_xx + dv_yy) + dp_y - fy) * 0.01
    loss3 = (du_x + dv_y) * 1e2
    # return loss1, loss2, loss3  # penalty
    return loss1, loss2, loss3, loss1, loss2, loss3  # augmented Lagrangian


def volume(inputs, outputs, X):
    return outputs[:, 3:4]


def loss_volume(_, y):
    return tf.math.square(tf.math.maximum(0.0, tf.reduce_mean(y) - GAMMA))


def dissipated_power(inputs, outputs, X):
    du = dde.grad.jacobian(outputs, inputs, i=0)
    dv = dde.grad.jacobian(outputs, inputs, i=1)
    p1 = tf.math.reduce_sum(
        tf.math.square(du) + tf.math.square(dv), axis=1, keepdims=True
    )
    u2 = tf.math.reduce_sum(tf.math.square(outputs[:, :2]), axis=1, keepdims=True)
    p2 = alpha(outputs[:, 3:]) * u2
    return 0.5 * (p1 + p2)


def loss_power(_, y):
    return tf.reduce_mean(y)


def output_transform(inputs, outputs):
    x, y = inputs[:, :1], inputs[:, 1:]
    bc = 16 * x * (1 - x) * y * (1 - y)

    # u
    u0 = 1
    u = tf.math.abs(u0 + bc * outputs[:, :1])
    # v
    v = bc * outputs[:, 1:2]
    # p
    p = (1 - x) * outputs[:, 2:3]
    # rho
    # rho = tf.math.exp(-bc * tf.math.square(outputs[:, 3:]))
    # rho = 1 + bc * outputs[:, 3:]
    center = tf.math.square(x - 0.5) + tf.math.square(y - 0.5)
    # rho = center * outputs[:, 3:]
    rho = center * (
        bc * outputs[:, 3:] + (1 - bc) * (1 + 1e-6 / 0.25) / (center + 1e-6)
    )
    rho = tf.math.maximum(0.0, tf.math.minimum(1.0, rho))
    return tf.concat((u, v, p, rho), axis=1)


def augmented_Lagrangian(model, geom, mu_PDE, mu_V, beta):
    x = model.data.train_x[np.sum(model.data.num_bcs) :]
    x_inside = model.data.train_x[: model.data.num_bcs[0]]
    lambla1 = np.zeros((len(x), 1))
    lambla2 = np.zeros((len(x), 1))
    lambla3 = np.zeros((len(x), 1))
    lambdaV = 0
    mus = [[mu_PDE, mu_V, lambdaV]]

    for i in range(1, 10):
        # lambla is 1/3 of that defined in the paper.
        residual1, residual2, residual3, _, _, _ = model.predict(x, operator=pde)
        lambla1 += 2 / 3 * mu_PDE * residual1
        lambla2 += 2 / 3 * mu_PDE * residual2
        lambla3 += 2 / 3 * mu_PDE * residual3
        dV = np.mean(model.predict(x_inside)[:, 3:4]) - GAMMA
        lambdaV = max(lambdaV + 2 * mu_V * dV, 0)

        mu_PDE *= beta
        mu_V *= beta
        mus.append([mu_PDE, mu_V, lambdaV])
        print("-" * 80)
        print(f"Iteration {i}: mu = {mu_PDE}, {mu_V}, lambdaV = {lambdaV}\n")

        def loss_PDE1(_, y):
            return tf.reduce_mean(lambla1 * y)

        def loss_PDE2(_, y):
            return tf.reduce_mean(lambla2 * y)

        def loss_PDE3(_, y):
            return tf.reduce_mean(lambla3 * y)

        def loss_V1(_, y):
            if lambdaV > 0:
                return tf.math.square(tf.reduce_mean(y) - GAMMA)
            return loss_volume(None, y)

        def loss_V2(_, y):
            return tf.reduce_mean(y) - GAMMA

        loss_weights = [mu_PDE / 3] * 3 + [1] * 3 + [mu_V, lambdaV, 1]
        loss = (
            ["MSE"] * 3
            + [loss_PDE1, loss_PDE2, loss_PDE3]
            + [loss_V1, loss_V2, loss_power]
        )
        model.compile("L-BFGS-B", loss=loss, loss_weights=loss_weights)
        losshistory, train_state = model.train(disregard_previous_best=True)

        np.savetxt(f"lambda1_{i}.dat", lambla1)
        np.savetxt(f"lambda2_{i}.dat", lambla2)
        np.savetxt(f"lambda3_{i}.dat", lambla3)
        np.savetxt(f"lambda_{i}.dat", np.hstack((x, lambla1, lambla2, lambla3)))
        np.savetxt("mu_lambdaV.dat", np.array(mus))
        save_solution(geom, model, f"solution{i}")


def main():
    geom = dde.geometry.Rectangle([0, 0], [1, 1])

    net = dde.maps.PFNN([2] + [[64] * 4] * 4 + [4], "tanh", "Glorot normal")  # ?
    net.apply_output_transform(output_transform)

    losses = [
        dde.OperatorBC(geom, volume, lambda x, _: not geom.on_boundary(x)),
        dde.OperatorBC(geom, volume, lambda x, _: not geom.on_boundary(x)),  # augmented Lagrangian
        dde.OperatorBC(geom, dissipated_power, lambda x, _: not geom.on_boundary(x)),
    ]

    dx = 0.01
    data = dde.data.PDE(
        geom,
        pde,
        losses,
        num_domain=int(geom.area / dx ** 2),
        num_boundary=int(geom.perimeter / dx),
    )
    model = dde.Model(data, net)

    mu_PDE, mu_V = 0.1, 1e4  # ?
    print("-" * 80)
    print(f"Iteration 0: mu = {mu_PDE}, {mu_V}\n")
    # loss_weights = [mu_PDE / 3] * 3 + [mu_V] + [1]  # penalty
    # loss = ["MSE", "MSE", "MSE", loss_volume, loss_power]
    loss_weights = [mu_PDE / 3] * 3 + [0] * 3 + [mu_V, 0] + [1]  # augmented Lagrangian
    loss = ["MSE"] * 3 + ["zero"] * 3 + [loss_volume, "zero", loss_power]
    model.compile(
        "adam", lr=0.0001, loss=loss, loss_weights=loss_weights,
    )
    losshistory, train_state = model.train(epochs=20000)
    # save_solution(geom, model, "solution")
    # return
    model.compile(
        "L-BFGS-B", loss=loss, loss_weights=loss_weights,
    )
    losshistory, train_state = model.train()
    save_solution(geom, model, "solution0")

    augmented_Lagrangian(model, geom, mu_PDE, mu_V, 2)

    dde.saveplot(losshistory, train_state, issave=True, isplot=False)


if __name__ == "__main__":
    main()
