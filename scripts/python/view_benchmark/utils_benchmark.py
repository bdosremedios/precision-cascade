import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from bokeh.palettes import Category20

cat20extract = Category20[20]

def load_benchmark_csv_us(
    data_dir: str, fname: str
) -> tuple[np.array, np.array]:

    data = np.loadtxt(
        os.path.join(data_dir, fname+".csv"),
        delimiter=",",
        ndmin=2,
        skiprows=1
    )

    return data[:, 0], data[:, 1]*1e-6

def load_dense_data(
    data_dir: str, fname: str
) -> tuple[np.array, np.array]:

    data_x, data_y = load_benchmark_csv_us(data_dir, fname)
    return data_x**2, data_y

def load_sparse_data(
    data_dir: str, fname: str
) -> tuple[np.array, np.array]:

    data_x, data_y = load_benchmark_csv_us(data_dir, fname)
    return data_x**1.5, data_y

def load_tri_dense_data(
    data_dir: str, fname: str
) -> tuple[np.array, np.array]:

    data_x, data_y = load_benchmark_csv_us(data_dir, fname)
    return 0.5*data_x**2, data_y

def load_tri_sparse_data(
    data_dir: str, fname: str
) -> tuple[np.array, np.array]:

    data_x, data_y = load_benchmark_csv_us(data_dir, fname)
    return 0.5*(data_x**1.5), data_y

def plot_data(
    ax: plt.Axes,
    data_dir: str,
    li_tup_fname_fmt_label: list[tuple[str, str, int, str]],
    load_data_func: Callable,
    disable_y: bool = False
) -> None:
    
    for fname, label, clridx, fmt in li_tup_fname_fmt_label:

        data_x, data_y = load_data_func(data_dir, fname)

        ax.plot(
            data_x, data_y, fmt,
            label=label,
            color=cat20extract[clridx]
        )

    if not disable_y:
        ax.set_ylabel("Seconds")
    ax.grid(True)
    ax.legend()