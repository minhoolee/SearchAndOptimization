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


def cross_entropy_method(f, x_init, sample_size, elite_ratio=0.2):
    x = np.array(x_init)
    f_list = [f(*x)]
    mu = x
    sigma = np.identity(n=len(x))
    num_elite = int(elite_ratio * sample_size)
    for i in range(MAX_EPOCHS):
        samples = np.random.multivariate_normal(mu, sigma, size=sample_size)
        f_samples = np.array([f(*sample) for sample in samples])
        elite_indices = f_samples.argsort()[:num_elite]
        elite_samples = samples[elite_indices]
        mu = np.mean(elite_samples, axis=0)
        sigma = np.cov(elite_samples, rowvar=False, ddof=0)
        f_average = np.mean(f_samples)
        f_list.append(f_average)
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
    sample_sizes = [10, 50]
    for seed in seeds:
        for i, sample_size in enumerate(sample_sizes):
            np.random.seed(seed)
            for j, (f, grad_f, x_init_f) in enumerate(f_data):
                f_list = cross_entropy_method(f, x_init_f, sample_size)
                axes[i][j].plot(np.arange(len(f_list)), f_list)
    for i in range(len(axes)):
        for j in range(len(axes[0])):
            axes[i][j].set_xlabel("Iterations")
            axes[i][j].set_ylabel(f"$f_{j + 1}$")
            axes[i][j].set_title(
                f"Cross Entropy Method For $f_{j + 1}$ With k = {sample_sizes[i]}, 5 Seeds"
            )
    plt.show()


if __name__ == "__main__":
    main()
