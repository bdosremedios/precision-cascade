import os
import re
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display

from bokeh.palettes import Category20

cat20 = Category20[20]

def check_file(file: str) -> None:

    if not os.path.isfile(file):
        raise RuntimeError(f"File {file} does not exist")

def check_dir(dir: str) -> None:

    if not os.path.isdir(dir):
        raise RuntimeError(f"Directory {dir} does not exist")
    
class Solver_ID_Info:

    fp_ids = {"FP64", "FP32", "FP16"}
    
    vp_ids = {
        "OuterRestartCount",
        "RelativeResidualThreshold",
        "CheckStagnation",
        "ThresholdToStagnation"
    }

    solver_color_fmt_dict = {
        "FP64": (cat20[0], ",-"),
        "FP32": (cat20[2], ",-"),
        "FP16": (cat20[4], ",-"),
        "OuterRestartCount": (cat20[6], ",-"),
        "RelativeResidualThreshold": (cat20[8], ",-"),
        "CheckStagnation": (cat20[10], ",-"),
        "ThresholdToStagnation": (cat20[12], ",-")
    }

class Individual_Experiment_Data:

    nan_r = r"(-nan|nan|-inf|inf)"

    def __scrub_nan(self, s: str) -> str:

        return re.sub(self.nan_r, "NaN", s)

    def __find_tol_iter(
            self, tol: float, relres_hist: np.array, inner_res_norm_hist: list[np.array]
        ) -> int:

        n = len(relres_hist)
        idx = -1
        for i in range(n):
            if relres_hist[n-i-1] > tol:
                idx = n-i-1
                break

        if idx == -1:
            raise RuntimeError(f"Tol idx not found in {self.json_path }")

        m = len(inner_res_norm_hist[idx])
        for i in range(m):
            if inner_res_norm_hist[idx][m-i-1] > tol:
                return self.inner_iter_hist[idx]+m-i

        return self.inner_iter_hist[idx]

    def __init__(self, json_path: str):

        file_in = open(json_path, "r")

        try:
            self.experiment_data = json.loads(self.__scrub_nan(file_in.read()))
        except json.JSONDecodeError as e:
            raise RuntimeError(
                "Read JSON error in " + json_path + " at " +
                f"line: {e.lineno} col: {e.colno}"
            )


        self.id = self.experiment_data["id"]
        self.json_path = json_path

        self.inner_iter_hist = (
            np.hstack(
                [[0], np.cumsum(self.experiment_data["inner_iterations"])]
            )
        )

        self.rel_res_history = (
            np.array(self.experiment_data["outer_res_norm_history"]) /
            self.experiment_data["outer_res_norm_history"][0]
        )

        norm_val = self.experiment_data["inner_res_norm_history"][0][0]
        self.inner_rel_res_history = [
            np.array(hist)/norm_val for hist in
            self.experiment_data["inner_res_norm_history"]
        ]

        tol = 1e-10
        self.iter_to_tol = -1
        if self.rel_res_history[-1] <= tol:
            self.iter_to_tol = self.__find_tol_iter(
                tol,
                self.rel_res_history,
                self.inner_rel_res_history
            )

    def print(self) -> None:

        print(
            f"id: {self.experiment_data['id']}\n" +
            f"solver-class : {self.experiment_data['solver_class']}\n" +
            f"initiated : {self.experiment_data['initiated']}\n" +
            f"converged : {self.experiment_data['converged']}\n" +
            f"terminated : {self.experiment_data['terminated']}\n" +
            f"elapsed-time-ms : {self.experiment_data['elapsed_time_ms']}\n" +
            f"relative-residual : {self.rel_res_history[-1]}\n" +
            f"outer-iterations : {self.experiment_data['outer_iterations']}\n"
            f"inner-iterations : {self.experiment_data['inner_iterations']}\n"
        )

    def plot_res_data(self, ax: plt.Axes, color: str, fmt: str) -> None:

        if self.id in Solver_ID_Info.vp_ids:
            ax.axvline(
                self.inner_iter_hist[
                    self.experiment_data["hlf_sgl_cascade_change"]-1
                ],
                linestyle=":",
                color=color
            )
            ax.axvline(
                self.inner_iter_hist[
                    self.experiment_data["sgl_dbl_cascade_change"]-1
                ],
                linestyle="-.",
                color=color
            )

        return ax.plot(
            self.inner_iter_hist,
            self.rel_res_history,
            fmt,
            color=color
        )
    
    def get_data_list(self):
        return [
            self.inner_iter_hist[-1],
            self.iter_to_tol,
            self.rel_res_history[-1],
            self.experiment_data["elapsed_time_ms"]
        ]

class Solver_Experiment_Data:

    def __init__(self, id: str, json_list: list[str]):

        self.id = id

        self.experiment_data = []
        for json in json_list:
            self.experiment_data.append(
                Individual_Experiment_Data(json)
            )
    
    def plot_res_data(self, ax: plt.Axes) -> None:

        first_to_label=True
        for experiment_data in self.experiment_data:
            lines = experiment_data.plot_res_data(
                ax, *(Solver_ID_Info.solver_color_fmt_dict[self.id])
            )
            if first_to_label:
                first_to_label = False
                lines[0].set_label(self.id)
    
    def generate_df_row_med(self) -> tuple[str, float, float, float, float]:

        arr = []
        for experiment_data in self.experiment_data:
            arr.append(experiment_data.get_data_list())
        arr = np.array(arr)

        return (
            self.id,
            np.median(arr[:, 0]),
            np.median(arr[:, 1]),
            np.median(arr[:, 2]),
            np.median(arr[:, 3])
        )


class Matrix_Experiment_Data:

    def __init__(
        self,
        matrix_id: str, matrix_dir: str, exp_iters: int, solver_ids: list[str]
    ):

        self.id = matrix_id

        # Check all iteration dirs exist in matrix dir
        iteration_dirs = [
            os.path.join(matrix_dir, str(i)) for i in range(exp_iters)
        ]
        for iter_dir in iteration_dirs:
            check_dir(iter_dir)

        # Load data grouped on solver
        self.solver_data = []
        for solver_id in solver_ids:

            # Check all solver data jsons exist in each iteration dir
            json_list = []
            for iter_dir in iteration_dirs:
                solver_json = os.path.join(iter_dir, solver_id+".json")
                check_file(solver_json)
                json_list.append(solver_json)

            self.solver_data.append(
                Solver_Experiment_Data(solver_id, json_list)
            )
    
    def generate_df_table(self) -> pd.DataFrame:

        fp64_idx = -1
        for i in range(len(self.solver_data)):
            if self.solver_data[i].id == "FP64":
                fp64_idx = i
        if fp64_idx == -1:
            raise RuntimeError("Missing FP64")


        in_order_data = []
        for solver_data in self.solver_data:
            in_order_data.append(list(solver_data.generate_df_row_med()))

        for row in in_order_data:
            row.append(row[3]/in_order_data[fp64_idx][3])
            row.append(row[4]/in_order_data[fp64_idx][4])

        df = pd.DataFrame(
            in_order_data,
            columns=[
                "Solver ID",
                "Inner Iterations",
                "1e-10 Inner Iteration",
                "Relative Residual",
                "Elapsed Time (ms)",
                "Relative Error",
                "Relative Time"
            ]
        )

        df = df.astype(
            {
                "Solver ID": 'string',
                "Inner Iterations": 'int32',
                "1e-10 Inner Iteration": 'int32',
                "Relative Residual": 'float64',
                "Elapsed Time (ms)": 'int32',
                "Relative Error": 'float64',
                "Relative Time": 'float64'
            }
        )

        return df

    def plot_fp_res_data(self, ax: plt.Axes) -> None:

        for solver_data in self.solver_data:
            if (solver_data.id in Solver_ID_Info.fp_ids):
                solver_data.plot_res_data(ax)

    def plot_vp_res_data(self, ax: plt.Axes) -> None:

        for solver_data in self.solver_data:
            if (solver_data.id in Solver_ID_Info.vp_ids):
                solver_data.plot_res_data(ax)

class Solve_Group_Data:

    def __init__(self, solve_group_dir: str):

        json_path = os.path.join(solve_group_dir, "solve_group_specs.json")
        check_file(json_path)
        self.solve_group_spec_data = json.load(open(json_path, "r"))

        self.matrix_experiment_data = []
        for matrix_id in self.solve_group_spec_data["matrix_ids"]:

            # Check matrices and iteration structures exist correctly
            matrix_dir = os.path.join(solve_group_dir, matrix_id)
            check_dir(matrix_dir)

            self.matrix_experiment_data.append(
                Matrix_Experiment_Data(
                    matrix_id,
                    matrix_dir,
                    self.solve_group_spec_data["experiment_iterations"],
                    self.solve_group_spec_data["solver_ids"]
                )
            )
    
    def display_data(self) -> None:

        for matrix_experiment_data in self.matrix_experiment_data:

            fig, axs = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)

            matrix_experiment_data.plot_fp_res_data(axs[0])
            axs[0].set_title("Fixed Precision Convergence")

            matrix_experiment_data.plot_vp_res_data(axs[1])
            axs[1].set_title("Adaptive Precision Convergence")

            axs[0].set_ylabel("$\\frac{|| b-Ax_{i}||_{2}}{||b-Ax_{0}||_{2}}$")

            for ax in axs:
                ax.set_xlabel("Inner Iterations")
                ax.semilogy()
                ax.grid()
                ax.legend()

            fig.suptitle(matrix_experiment_data.id)
            fig.tight_layout()

            plt.show()

            display(matrix_experiment_data.generate_df_table())
