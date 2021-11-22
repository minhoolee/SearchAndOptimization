import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


MAX_EPOCHS = 100


def f1(x_1, x_2):
    return x_1 ** 2 + x_2 ** 2


def grad_f1(x_1, x_2):
    return np.array([2 * x_1, 2 * x_2])


def f2(x_1, x_2):
    return -1 * (
        (1 + np.cos(12 * np.sqrt(x_1 ** 2 + x_2 ** 2)))
        / (0.5 * (x_1 ** 2 + x_2 ** 2) + 2)
    )


def grad_f2(x_1, x_2):
    a = np.sqrt(x_1 ** 2 + x_2 ** 2)
    shared_term = (
        12 * (0.5 * a ** 2 + 2) * np.sin(12 * a) / a + np.cos(12 * a) + 1
    ) / (0.5 * a ** 2 + 2) ** 2
    return np.array([x_1 * shared_term, x_2 * shared_term,])


def f3(*xs):
    assert len(xs) == 50
    return sum(x_i ** 2 for x_i in xs)


def grad_f3(*xs):
    assert len(xs) == 50
    return np.array([2 * x_i for x_i in xs])


def simulated_annealing(f, x_init, temperature):
    x = np.array(x_init)
    curr_f = f(*x)
    f_list = [curr_f]
    mu = np.zeros(x.shape)
    sigma = np.identity(n=len(x))
    for k in range(1, MAX_EPOCHS + 1):
        T_k = temperature / k
        dx = np.random.multivariate_normal(mu, sigma)
        new_f = f(*(x + dx))
        if new_f < curr_f or np.random.rand() < np.exp((curr_f - new_f) / T_k):
            x = x + dx
            curr_f = new_f
        f_list.append(curr_f)
    return f_list


def main():
    x_init = 2
    alpha = 0.01
    x_init_f1 = np.array([x_init, x_init])
    x_init_f2 = np.array([x_init, x_init])
    x_init_f3 = np.array([x_init for _ in range(50)])
    f_data = [
        (f1, grad_f1, x_init_f1),
        (f2, grad_f2, x_init_f2),
        (f3, grad_f3, x_init_f3),
    ]
    seeds = [42, 187372311, 204110176, 129995678, 6155814]

    fig, axes = plt.subplots(2, 3, figsize=(20, 15))
    temperatures = [1000, 10]
    for seed in seeds:
        for i, temperature in enumerate(temperatures):
            np.random.seed(seed)
            for j, (f, grad_f, x_init_f) in enumerate(f_data):
                f_list = simulated_annealing(f, x_init_f, temperature)
                axes[i][j].plot(np.arange(len(f_list)), f_list)
    for i in range(len(axes)):
        for j in range(len(axes[0])):
            axes[i][j].set_xlabel("Iterations")
            axes[i][j].set_ylabel(f"$f_{j + 1}$")
            axes[i][j].set_title(
                f"Simulated Annealing For $f_{j + 1}$ With T = {temperatures[i]}, 5 Seeds"
            )
    plt.show()


if __name__ == "__main__":
    main()
