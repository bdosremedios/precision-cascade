import json
import os
import numpy as np
import matplotlib.pyplot as plt
from bokeh.palettes import Category20

cat20 = Category20[20]

def check_file(file):

    if not os.path.isfile(file):
        raise RuntimeError(f"File {file} does not exist")

def check_dir(dir):

    if not os.path.isdir(dir):
        raise RuntimeError(f"Directory {dir} does not exist")
    
class Solver_ID_Info:

    fp_ids = {"FP64", "FP32", "FP16"}
    
    mp_ids = {
        "OuterRestartCount",
        "RelativeResidualThreshold",
        "CheckStagnation",
        "ProjectThresholdAfterStagnation"
    }

    solver_color_fmt_dict = {
        "FP64": (cat20[0], ",-"),
        "FP32": (cat20[2], ",-"),
        "FP16": (cat20[4], ",-"),
        "OuterRestartCount": (cat20[6], ",-"),
        "RelativeResidualThreshold": (cat20[8], ",-"),
        "CheckStagnation": (cat20[10], ",-"),
        "ProjectThresholdAfterStagnation": (cat20[12], ",-")
    }

class Individual_Experiment_Data:

    def __scrub_nan(self, s):
        return s.replace("-nan", "NaN").replace("nan", "NaN")

    def __init__(self, json_path):

        file_in = open(json_path, "r")
        self.experiment_data = json.loads(self.__scrub_nan(file_in.read()))

        self.id = self.experiment_data["id"]

        self.rel_res_history = (
            np.array(self.experiment_data["outer_res_norm_history"]) /
            self.experiment_data["outer_res_norm_history"][0]
        )

    def print(self):

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

    def plot_res_data(self, ax, color, fmt):

        if self.id in Solver_ID_Info.mp_ids:
            ax.axvline(
                self.experiment_data["hlf_sgl_cascade_change"],
                linestyle=":",
                color=color
            )
            ax.axvline(
                self.experiment_data["sgl_dbl_cascade_change"],
                linestyle="--",
                color=color
            )

        return ax.plot(
            np.arange(0, self.experiment_data["outer_iterations"]+1, 1),
            self.rel_res_history,
            fmt,
            color=color
        )

class Solver_Experiment_Data:

    def __init__(self, id, json_list):

        self.id = id

        self.experiment_data = []
        for json in json_list:
            self.experiment_data.append(
                Individual_Experiment_Data(json)
            )
    
    def plot_res_data(self, ax):

        first_to_label=True
        for experiment_data in self.experiment_data:
            lines = experiment_data.plot_res_data(
                ax, *(Solver_ID_Info.solver_color_fmt_dict[self.id])
            )
            if first_to_label:
                first_to_label = False
                lines[0].set_label(self.id)

class Matrix_Experiment_Data:

    def __init__(self, matrix_id, matrix_dir, exp_iters, solver_ids):

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

    def plot_fp_res_data(self, ax):

        for solver_data in self.solver_data:
            if (solver_data.id in Solver_ID_Info.fp_ids):
                solver_data.plot_res_data(ax)

    def plot_mp_res_data(self, ax):

        for solver_data in self.solver_data:
            if (solver_data.id in Solver_ID_Info.mp_ids):
                solver_data.plot_res_data(ax)

class Solve_Group_Data:

    def __init__(self, solve_group_dir):

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
    
    def display_data(self):

        for matrix_experiment_data in self.matrix_experiment_data:

            fig, axs = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)

            matrix_experiment_data.plot_fp_res_data(axs[0])
            axs[0].set_title("Fixed Precision Convergence")

            matrix_experiment_data.plot_mp_res_data(axs[1])
            axs[1].set_title("Mixed Precision Convergence")

            axs[0].set_ylabel("$\\frac{|| b-Ax_{i}||_{2}}{||b-Ax_{0}||_{2}}$")

            for ax in axs:
                ax.set_xlabel("Outer Iterations")
                ax.semilogy()
                ax.grid()
                ax.legend()

            fig.suptitle(matrix_experiment_data.id)
            fig.tight_layout()

            plt.show()
