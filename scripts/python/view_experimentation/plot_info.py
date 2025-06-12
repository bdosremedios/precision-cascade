from exp_spec_info import (
    SETUPS,
    SETUP_MATRIX_MAPPING,
    RESTART_PARAMS,
    ALL_MATRICES,
    SMALL_MATRICES,
    ILU0_MATRICES
)
from bokeh.palettes import Category20

cat20 = Category20[20]

SOLVER_CLR_DICT = {
    "FP FP64": "#CC6677",
    "FP FP32": "#332288",
    "FP FP16": "#DDCC77",
    "PC HSD ORC": "#117733",
    "PC HSD RRT": "#88CCEE",
    "PC HSD CS": "#882255",
    "PC HSD S2T": "#44AA99",
    "PC SD ORC": "#999933",
    "PC SD RRT": "#AA4499",
    "PC SD CS": "#000000",
}

SOLVER_FMT_DICT = {
    "FP FP64": ",-",
    "FP FP32": ",-",
    "FP FP16": ",-",
    "PC HSD ORC": ",-",
    "PC HSD RRT": ",-",
    "PC HSD CS": ",-",
    "PC HSD S2T": ",-",
    "PC SD ORC": ",-",
    "PC SD RRT": ",-",
    "PC SD CS": ",-",
}

SETUP_NAME_MAPPING = {
    "ilu0": "ILU(0)",
    "ilutp1em2": "ILUTP($20$, $10^{-2}$)",
    "ilutp1em4": "ILUTP($20$, $10^{-4}$)",
    "unpreconddense": "Unpreconditioned Dense",
    "unprecond": "Unpreconditioned Sparse"
}
SOLVER_ID_ORDER = [
    "FP FP16", "FP FP32", "FP FP64",
    "PC HSD ORC", "PC HSD CS", "PC HSD RRT", "PC HSD S2T",
    "PC SD ORC", "PC SD CS", "PC SD RRT"
]

UNIQUE_SETUP_ENUMERATION = {}
index = 0
for setup in SETUPS:
    for matrix in SETUP_MATRIX_MAPPING[setup]:
        for restart_param in RESTART_PARAMS:
            UNIQUE_SETUP_ENUMERATION[(setup, matrix, restart_param)] = index
            index += 1


def plot_exp_iters_conv_traj(ax, solver_exp_iteration_data, n_iterations, label, color):
    first = True
    for i in range(n_iterations):
        plot_data = solver_exp_iteration_data[
            solver_exp_iteration_data["experiment_iter"] == i
        ]
        if len(plot_data) > 0:
            kwargs = {
                "color": color,
                "label": label if first else None
            }
            ax.semilogy(plot_data.iloc[0]["inner_relres"], **kwargs)
            first = False