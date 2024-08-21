import json
import numpy as np
import matplotlib.pyplot as plt

class Solve_Data:

    def __scrub_nan(self, s):
        return s.replace("-nan", "NaN")

    def __init__(self, json_path):

        file_in = open(json_path, "r")
        file_json = json.loads(self.__scrub_nan(file_in.read()))
        
        self.id = file_json["id"]
        self.solver_class = file_json["solver_class"]

        self.initiated = file_json["initiated"]
        self.converged = file_json["converged"]
        self.terminated = file_json["terminated"]

        self.elapsed_time_ms = file_json["elapsed_time_ms"]

        self.outer_iterations = file_json["outer_iterations"]
        self.inner_iterations = np.array(file_json["inner_iterations"])

        self.outer_res_norm_history = file_json["outer_res_norm_history"]
        self.inner_res_norm_history = file_json["inner_res_norm_history"]

        self.rel_res_history = (
            np.array(self.outer_res_norm_history) /
            self.outer_res_norm_history[0]
        )

    def print(self):

        print(
            f"id: {self.id}\n" +
            f"solver-class : {self.solver_class}\n" +
            f"initiated : {self.initiated}\n" +
            f"converged : {self.converged}\n" +
            f"terminated : {self.terminated}\n" +
            f"elapsed-time-ms : {self.elapsed_time_ms}\n" +
            f"relative-residual : {self.outer_res_norm_history[-1]/self.outer_res_norm_history[0]}\n" +
            f"outer-iterations : {self.outer_iterations}\n"
            f"inner-iterations : {self.inner_iterations}\n"
        )

    def plot_res_data(self, ax, fmt):

        ax.plot(
            np.arange(0, self.outer_iterations+1, 1),
            self.rel_res_history,
            fmt, label=self.id
        )

# class SolveGroup