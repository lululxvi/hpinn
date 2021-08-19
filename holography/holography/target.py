import deepxde as dde

from .config import getE, target


def target_bc(inputs, outputs, X):
    return getE(outputs, 3, "Re") ** 2 + getE(outputs, 3, "Im") ** 2 - target(X)


class TargetErr(dde.callbacks.Callback):
    def __init__(self, geom, period=1, filename=None):
        super(TargetErr, self).__init__()
        self.period = period

        self.x = geom.uniform_points(50000)
        self.y_true = target(self.x)
        self.file = sys.stdout if filename is None else open(filename, "w", buffering=1)
        self.value = None
        self.epochs_since_last = 0

    def init(self):
        self.feed_dict = self.model.net.feed_dict(False, False, 2, self.x)

    def on_train_begin(self):
        y_pred = self.model.sess.run(self.model.net.outputs, feed_dict=self.feed_dict)
        y_pred = (y_pred[:, 0] ** 2 + y_pred[:, 1] ** 2) ** 0.5
        self.value = np.mean((self.y_true - y_pred) ** 2)
        print(
            self.model.train_state.epoch, self.value, file=self.file,
        )
        self.file.flush()

    def on_train_end(self):
        self.on_train_begin()

    def on_epoch_end(self):
        self.epochs_since_last += 1
        if self.epochs_since_last >= self.period:
            self.epochs_since_last = 0
            self.on_train_begin()


# def target1(inputs, outputs, X):
#     return getE(outputs, 3, "Re") ** 2 + getE(outputs, 3, "Im") ** 2 - 1


# def target0(inputs, outputs, X):
#     return getE(outputs, 3, "Re") ** 2 + getE(outputs, 3, "Im") ** 2
