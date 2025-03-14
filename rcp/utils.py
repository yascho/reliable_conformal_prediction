import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def step_plot(data, xticks, xticklabels, xlabel, ylabel, title):
    sns.set_style("whitegrid")
    sns.set_context("notebook")
    fig, ax = plt.subplots()

    plt.xticks(xticks)
    plt.xlim(0, max(xticks))

    for key in data.keys():
        plt.step(xticks, data[key][:len(xticks)], label=key, where="post")

    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.xaxis.set_minor_locator(ticker.FixedLocator(xticks+0.5))
    ax.xaxis.set_minor_formatter(ticker.FixedFormatter(xticklabels))

    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
