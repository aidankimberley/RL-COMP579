# Plotting utilities for COMP579 Assignment 3.
# Compare related experiments (e.g., different hyperparameters) in one plot.

import numpy as np
import matplotlib.pyplot as plt


def plot_compare_smoothed_rewards(
    train_rewards_list,
    labels=None,
    window=10,
    xlabel="Episode",
    ylabel="Discounted Reward",
    title=None,
    save_path=None,
):
    """
    Plot comparison of multiple reward curves (mean ± std across seeds).

    Args:
        train_rewards_list : list[list[list[float]]]
            Outer list over methods.
            Each element is a list over seeds,
            where each seed is a reward sequence.

            Example:
                [
                    [[...], [...], ...],   # method 1 seeds
                    [[...], [...], ...],   # method 2 seeds
                    ...
                ]

        labels : list[str] or None
            Labels for each curve. If None, default names are used.

        window : int
            Moving-average smoothing window.

        xlabel : str
        ylabel : str
    """

    if labels is None:
        labels = [f"Method {i+1}" for i in range(len(train_rewards_list))]

    if len(labels) != len(train_rewards_list):
        raise ValueError("labels must match number of methods")

    def compute_stats(train_rewards):
        smoothed = [
            np.convolve(r, np.ones(window) / window, mode="valid")
            for r in train_rewards
        ]
        smoothed = np.array(smoothed)

        avg = np.mean(smoothed, axis=0)
        std = np.std(smoothed, axis=0)
        return avg, std

    stats = [compute_stats(tr) for tr in train_rewards_list]

    # align lengths across methods
    min_len = min(len(avg) for avg, _ in stats)
    x = np.arange(min_len)

    for (avg, std), label in zip(stats, labels):
        avg = avg[:min_len]
        std = std[:min_len]

        plt.plot(x, avg, label=label)
        plt.fill_between(x, avg - std, avg + std, alpha=0.3)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    plt.close()