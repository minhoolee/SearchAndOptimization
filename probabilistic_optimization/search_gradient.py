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


def search_gradient(f, x_init, sample_size, normalize_gradients=False, lr=0.01):
    x = np.array(x_init)
    f_list = [f(*x)]
    mu = x
    sigma = np.identity(n=len(x))
    last_sigma = sigma
    update_sigma = True
    for i in range(MAX_EPOCHS):
        try:
            samples = np.random.multivariate_normal(
                mu, sigma, size=sample_size, check_valid="raise"
            )
        except ValueError as e:
            update_sigma = False
            sigma = last_sigma
        grad_expectation_mu = 0
        grad_expectation_sigma = 0
        f_samples = []
        for i in range(sample_size):
            f_sample = f(*samples[i])
            f_samples.append(f_sample)
            sigma_inv = np.linalg.inv(sigma)
            log_derivative_mu = sigma_inv @ (samples[i] - mu)
            log_derivative_sigma = (
                -(1 / 2) * sigma_inv
                + (1 / 2)
                * sigma_inv
                @ (samples[i] - mu).reshape(-1, 1)
                @ (samples[i] - mu).reshape(1, -1)
                @ sigma_inv
            )
            grad_expectation_mu += f_sample * log_derivative_mu
            grad_expectation_sigma += f_sample * log_derivative_sigma
        grad_expectation_mu /= sample_size
        grad_expectation_sigma /= sample_size
        # Normalize gradients to prevent exploding gradients
        if normalize_gradients:
            grad_expectation_mu /= np.linalg.norm(grad_expectation_mu)
            grad_expectation_sigma /= np.linalg.norm(grad_expectation_sigma)
        mu = mu - lr * grad_expectation_mu
        if update_sigma:
            # Save last sigma to use in case the updated sigma no longer remains positive semi-definite
            last_sigma = sigma
            sigma = sigma - lr * grad_expectation_sigma
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
                f_list = search_gradient(
                    f, x_init_f, sample_size, normalize_gradients=(j == 2), lr=0.01
                )
                axes[i][j].plot(np.arange(len(f_list)), f_list)
    for i in range(len(axes)):
        for j in range(len(axes[0])):
            axes[i][j].set_xlabel("Iterations")
            axes[i][j].set_ylabel(f"$f_{j + 1}$")
            if j == 2:
                axes[i][j].set_title(
                    f"(Normalized) Search Gradient For $f_{j + 1}$ With k = {sample_sizes[i]}, 5 Seeds"
                )
            else:
                axes[i][j].set_title(
                    f"Search Gradient For $f_{j + 1}$ With k = {sample_sizes[i]}, 5 Seeds"
                )
    plt.show()


if __name__ == "__main__":
    main()
