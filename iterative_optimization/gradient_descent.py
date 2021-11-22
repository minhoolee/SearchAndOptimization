import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera

MAX_ITERS = 25
ALPHA = 0.3


def f(x_1, x_2):
    return (x_1 ** 2) - (x_1 * x_2) + 3 * (x_2 ** 2) + 5


def grad_f(x_1, x_2):
    # [df/d{x_1}, df/d{x_2}]
    return np.array([(2 * x_1) - x_2, -x_1 + (6 * x_2)])


def hessian_f(a, b):
    # [d^2f/d{x_1}^2, d^2f/d{x_2}^2]
    return np.array([[2, -1], [6, -1]])


def gradient_descent(f, grad_f, x_init, alpha):
    x = np.array(x_init)
    p = -1 * grad_f(*x)
    f_list = [f(*x)]
    x_list = [x]
    dir_list = [p]
    while not np.all(np.isclose(p, 0)):
        x = (x[0] + (alpha * p[0]), x[1] + (alpha * p[1]))
        p = -1 * grad_f(*x)
        f_list.append(f(*x))
        x_list.append(x)
        dir_list.append(p)
    return f_list, x_list, dir_list


def main():
    x_init = (2, 2)

    f_list, x_list, dir_list = gradient_descent(f, grad_f, x_init, alpha=ALPHA)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    camera = Camera(fig)
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("$f(x_1, x_2)$")
    ax1.set_title("Function Values")
    ax2.set_xlabel("$x_1$")
    ax2.set_ylabel("$x_2$")
    ax2.set_title("Gradients")
    ax2.set_xlim(-1, 4)
    ax2.set_ylim(-1, 4)
    fig.suptitle(f"Gradient Descent Using Fixed Step $\\alpha = {ALPHA}$")

    n = min(MAX_ITERS, len(f_list))
    xx, yy = np.meshgrid(np.linspace(*ax2.get_xlim()), np.linspace(*ax2.get_ylim()))
    zz = f(xx, yy)
    x, y = zip(*x_list[:n])
    dx, dy = zip(*dir_list[:n])
    for i in range(n):
        ax1.plot(np.arange(i + 1), f_list[: (i + 1)], "-b.")
        contours = ax2.contour(xx, yy, zz, levels=20)
        ax2.quiver(
            x[:i],
            y[:i],
            dx[:i],
            dy[:i],
            angles="xy",
            scale_units="xy",
            scale=(1 / ALPHA),
            color="black",
        )
        ax2.quiver(
            x[i],
            y[i],
            dx[i],
            dy[i],
            angles="xy",
            scale_units="xy",
            scale=(1 / ALPHA),
            color="red",
        )
        camera.snap()
    animation = camera.animate(interval=1000)
    animation.save("gradient_descent.gif", writer="imagemagick")


if __name__ == "__main__":
    main()
