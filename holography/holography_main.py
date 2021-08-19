from scipy.interpolate import griddata

import deepxde as dde
import numpy as np
from deepxde.backend import tf

# tf.random.set_random_seed(1234)

from holography.bc import feature_transform, output_transform
from holography.config import BOX, DPML, J
from holography.pde import pde_domain
from holography.target import target_bc


def l2_relative_error_1(y_true, y_pred):
    return dde.metrics.nanl2_relative_error(y_true[:, 0], y_pred[:, 0])


def l2_relative_error_2(y_true, y_pred):
    return dde.metrics.nanl2_relative_error(y_true[:, 1], y_pred[:, 1])


def solution_forward(x):
    # solution for normal distribution J
    d = np.loadtxt("FDTD.dat")
    ReE = griddata(d[:, :2], d[:, 2], x)
    ImE = griddata(d[:, :2], d[:, 3], x)
    return np.vstack((ReE, ImE)).T


def save_solution(geom, model, filename_E, filename_residual):
    # x_center = np.vstack(
    #     (np.full(1000, 0), np.linspace(BOX[0][1] - DPML, BOX[1][1] + DPML, num=1000))
    # ).T
    # y_pred = model.predict(x_center)
    # print("Saving E_x0.dat ...")
    # np.savetxt("E_x0.dat", np.hstack((x_center, y_pred)))
    # print("Saving J.dat ...")
    # np.savetxt("J.dat", np.hstack((x_center, J(x_center))))

    x = geom.uniform_points(50000)
    y_pred = model.predict(x)
    print("Saving E ...\n")
    np.savetxt(filename_E, np.hstack((x, y_pred[:, :2])))

    # residual_Re, residual_Im = model.predict(x, operator=pde_domain)
    residual_Re, residual_Im, _, _ = model.predict(x, operator=pde_domain)
    print("Saving residual ...\n")
    np.savetxt(filename_residual, np.hstack((x, residual_Re, residual_Im)))


def save_epsilon(geom, model, filename):
    x = geom.uniform_points(60000)
    y_pred = model.predict(x)
    print("Saving epsilon ...\n")
    np.savetxt(filename, np.hstack((x, y_pred[:, -1:])))


def penalty(model, geom2, mu, beta):
    i = 0
    while mu < 100:
        i += 1
        mu *= beta
        print("-" * 80)
        print(f"Iteration {i}: mu = {mu}\n")

        loss_weights = [0.5 * mu] * 2 + [1]
        # model.compile("adam", lr=0.001, loss_weights=loss_weights)
        # losshistory, train_state = model.train(epochs=1000)
        model.compile("L-BFGS-B", loss_weights=loss_weights)
        losshistory, train_state = model.train(disregard_previous_best=True)

        save_epsilon(geom2, model, f"epsilon{i}.dat")
        # save_solution(geom, model, f"E{i}.dat", f"residual{i}.dat")


def augmented_Lagrangian(model, geom, geom2, mu, beta):
    x = model.data.train_x[np.sum(model.data.num_bcs) :]
    lambla_Re, lambla_Im = np.zeros((len(x), 1)), np.zeros((len(x), 1))

    for i in range(1, 10):
        # lambla_Re and lambla_Im are one half smaller than that defined in the paper.
        residual_Re, residual_Im, _, _ = model.predict(x, operator=pde_domain)
        lambla_Re += mu * residual_Re
        lambla_Im += mu * residual_Im

        mu *= beta
        print("-" * 80)
        print(f"Iteration {i}: mu = {mu}\n")

        def loss_Lagrangian_Re(_, y):
            return tf.reduce_mean(lambla_Re * y)

        def loss_Lagrangian_Im(_, y):
            return tf.reduce_mean(lambla_Im * y)

        loss_weights = [0.5 * mu] * 2 + [1, 1] + [1]
        loss = ["MSE", "MSE", loss_Lagrangian_Re, loss_Lagrangian_Im, "MSE"]
        model.compile("L-BFGS-B", loss=loss, loss_weights=loss_weights)
        losshistory, train_state = model.train(disregard_previous_best=True)

        save_epsilon(geom2, model, f"epsilon{i}.dat")
        np.savetxt(f"lambda_Re{i}.dat", lambla_Re)
        np.savetxt(f"lambda_Im{i}.dat", lambla_Im)
        np.savetxt(f"lambda{i}.dat", np.hstack((x, lambla_Re, lambla_Im)))
        # save_solution(geom, model, f"E{i}.dat", f"residual{i}.dat")


def main():
    # In some GPUs, float64 is required to make L-BFGS work for some reason... 
    # dde.config.real.set_float64()

    geom = dde.geometry.Rectangle(BOX[0] - DPML, BOX[1] + DPML)
    geom1 = dde.geometry.Rectangle(BOX[0] - DPML, [BOX[1][0] + DPML, -1])
    geom2 = dde.geometry.Rectangle([BOX[0][0] - DPML, -1], [BOX[1][0] + DPML, 0])
    geom3 = dde.geometry.Rectangle([BOX[0][0] - DPML, 0], BOX[1] + DPML)
    geom3_small = dde.geometry.Rectangle([BOX[0][0], 0], BOX[1])
    # geom3_in = dde.geometry.Rectangle([-0.5, 1], [0.5, 2])
    # geom3_out = geom3 - geom3_in

    net = dde.maps.PFNN([2] + [[48] * 3] * 4 + [3], "tanh", "Glorot normal")
    net.apply_feature_transform(feature_transform)
    net.apply_output_transform(output_transform)

    # Fit to the planewave solution
    # E0 = np.loadtxt("E0_normal.dat")
    # E0 = E0[np.random.choice(len(E0), size=10000, replace=False)]
    # ptset = dde.bc.PointSet(E0[:, :2])
    # inside = lambda x, _: ptset.inside(x)
    # loss0 = [
    #     dde.DirichletBC(geom, ptset.values_to_func(E0[:, 2:3]), inside, component=0),
    #     dde.DirichletBC(geom, ptset.values_to_func(E0[:, 3:4]), inside, component=1),
    #     dde.DirichletBC(geom, ptset.values_to_func(E0[:, 4:5]), inside, component=2),
    # ]

    # data = dde.data.PDE(geom, pde_domain, loss0, anchors=E0[:, :2])
    # model = dde.Model(data, net)
    # checkpointer = dde.callbacks.ModelCheckpoint(
    #     "model/model.ckpt", verbose=1, save_better_only=True
    # )
    # model.compile("adam", lr=0.001, loss_weights=[0] * 2 + [10] * 2 + [0.1])
    # losshistory, train_state = model.train(epochs=3000)
    # model.compile("adam", lr=0.001, loss_weights=[1] * 2 + [10] * 2 + [0])
    # losshistory, train_state = model.train(
    #     epochs=10000, callbacks=[checkpointer], disregard_previous_best=True
    # )
    # model.compile("L-BFGS-B", loss_weights=[1] * 2 + [10] * 2 + [0])
    # losshistory, train_state = model.train(callbacks=[checkpointer])
    # dde.saveplot(losshistory, train_state, issave=True, isplot=False)
    # save_solution(geom, model)
    # return

    losses = []
    # PDE (3)
    # losses += [
    #     dde.OperatorBC(geom, pde_E1_ReIm, lambda x, _: geom1.inside(x)),
    #     dde.OperatorBC(geom, pde_E2_ReIm, lambda x, _: geom2.inside(x)),
    #     dde.OperatorBC(geom, pde_E3_ReIm, lambda x, _: geom3.inside(x)),
    # ]

    # PML BC
    # Periodic BC (12)
    # losses += [
    #     dde.PeriodicBC(geom, 0, boundary1_leftright, derivative_order=0, component=0),
    #     dde.PeriodicBC(geom, 0, boundary1_leftright, derivative_order=1, component=0),
    #     dde.PeriodicBC(geom, 0, boundary1_leftright, derivative_order=0, component=1),
    #     dde.PeriodicBC(geom, 0, boundary1_leftright, derivative_order=1, component=1),
    #     dde.PeriodicBC(geom, 0, boundary2, derivative_order=0, component=2),
    #     dde.PeriodicBC(geom, 0, boundary2, derivative_order=1, component=2),
    #     dde.PeriodicBC(geom, 0, boundary2, derivative_order=0, component=3),
    #     dde.PeriodicBC(geom, 0, boundary2, derivative_order=1, component=3),
    #     dde.PeriodicBC(geom, 0, boundary3_leftright, derivative_order=0, component=4),
    #     dde.PeriodicBC(geom, 0, boundary3_leftright, derivative_order=1, component=4),
    #     dde.PeriodicBC(geom, 0, boundary3_leftright, derivative_order=0, component=5),
    #     dde.PeriodicBC(geom, 0, boundary3_leftright, derivative_order=1, component=5),
    # ]
    # Dirichlet BC (4)
    # losses += [
    #     dde.DirichletBC(geom, lambda _: 0, boundary1_bottom, component=0),
    #     dde.DirichletBC(geom, lambda _: 0, boundary1_bottom, component=1),
    #     dde.DirichletBC(geom, lambda _: 0, boundary3_top, component=4),
    #     dde.DirichletBC(geom, lambda _: 0, boundary3_top, component=5),
    # ]

    # Interface between Omega_1 and Omega_2 (4)
    # losses += [
    #     dde.OperatorBC(geom, interface12_ReE, interface12),
    #     dde.OperatorBC(geom, interface12_ImE, interface12),
    #     dde.OperatorBC(geom, interface12_dReE, interface12),
    #     dde.OperatorBC(geom, interface12_dImE, interface12),
    # ]
    # Interface between Omega_2 and Omega_3 (4)
    # losses += [
    #     dde.OperatorBC(geom, interface23_ReE, interface23),
    #     dde.OperatorBC(geom, interface23_ImE, interface23),
    #     dde.OperatorBC(geom, interface23_dReE, interface23),
    #     dde.OperatorBC(geom, interface23_dImE, interface23),
    # ]

    # Target (2)
    losses += [
        dde.OperatorBC(geom, target_bc, lambda x, _: geom3_small.inside(x)),
        # dde.OperatorBC(geom, target1, lambda x, _: geom3_in.inside(x)),
        # dde.OperatorBC(geom, target0, lambda x, _: geom3_out.inside(x)),
    ]

    # Points
    dx = 0.05
    # Extra points
    # h = 0.6
    # g = dde.geometry.Rectangle(
    #     [BOX[0][0] - DPML, -1.5 - h], [BOX[1][0] + DPML, -1.5 + h]
    # )
    # anchors = g.random_points(int(g.area / (dx / 4) ** 2))

    data = dde.data.PDE(
        geom,
        pde_domain,
        # None,
        losses,
        # [],
        num_domain=int(geom.area / dx ** 2),
        num_boundary=int(geom.perimeter / dx),
        # anchors=anchors,
        # num_test=50000,
        # solution=solution_forward,
    )
    model = dde.Model(data, net)

    # loss_weights = [0.5] * 2
    mu = 2
    print("-" * 80)
    print(f"Iteration 0: mu = {mu}\n")
    # loss_weights = [0.5 * mu] * 2 + [1]  # penalty
    loss_weights = [0.5 * mu] * 2 + [0, 0] + [1]  # augmented_Lagrangian
    model.compile(
        "adam",
        lr=0.001,
        loss_weights=loss_weights,
        # metrics=[l2_relative_error_1, l2_relative_error_2],
    )
    losshistory, train_state = model.train(epochs=20000)
    # save_epsilon(geom2, model, "epsilon_init.dat")
    # return
    model.compile(
        "L-BFGS-B",
        loss_weights=loss_weights,
        # metrics=[l2_relative_error_1, l2_relative_error_2],
    )
    losshistory, train_state = model.train()
    save_epsilon(geom2, model, "epsilon0.dat")
    # save_solution(geom, model, "E0.dat", "residual0.dat")

    # penalty(model, geom2, mu, 2)
    augmented_Lagrangian(model, geom, geom2, mu, 2)

    dde.saveplot(losshistory, train_state, issave=True, isplot=False)


if __name__ == "__main__":
    main()
