import numpy as np
import matplotlib.pyplot as plt


def plot_compare_smoothed_rewards(
    train_rewards_list,
    labels=None,
    window=50,
    xlabel="Episode",
    ylabel="Average Reward",
    title=None,
):
    """
    Clean, report-ready comparison plot (mean ± std).
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

    # Align lengths across methods
    min_len = min(len(avg) for avg, _ in stats)
    x = np.arange(min_len)

    plt.figure(figsize=(8, 5))

    for (avg, std), label in zip(stats, labels):
        avg = avg[:min_len]
        std = std[:min_len]

        plt.plot(x, avg, linewidth=2.5, label=label)
        plt.fill_between(x, avg - std, avg + std, alpha=0.2)

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    if title is not None:
        plt.title(title, fontsize=13)

    plt.legend(frameon=False)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()