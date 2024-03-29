#ifndef EXPERIMENT_RUN_H
#define EXPERIMENT_RUN_H

#include <filesystem>
#include <format>
#include <string>
#include <utility>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "types/types.h"
#include "tools/read_matrix.h"

#include "experiment_log.h"
#include "experiment_read.h"
#include "experiment_record.h"

#include "solvers/IterativeSolve.h"
#include "solvers/nested/GMRES_IR/MP_GMRES_IR.h"
#include "solvers/nested/GMRES_IR/FP_GMRES_IR.h"

namespace fs = std::filesystem;

static const double u_hlf = std::pow(2, -10);
static const double u_sgl = std::pow(2, -23);
static const double u_dbl = std::pow(2, -52);

template <template <typename> typename M>
using LinSysSolnPair = std::pair<TypedLinearSystem<M, double>, Vector<double>>;

template <template <typename> typename M>
LinSysSolnPair<M> load_linear_problem(
    cublasHandle_t handle,
    fs::path input_dir,
    std::string matrix_name,
    Experiment_Log logger
) {

    fs::path matrix_path = input_dir / fs::path(matrix_name+".csv");

    logger.info(std::format("Loading: {}", matrix_path.string()));

    M<double> A(read_matrixCSV<M, double>(handle, matrix_path));
    A.normalize_magnitude();
    logger.info(std::format("Matrix info: {}", A.get_info_string()));

    Vector<double> true_x(Vector<double>::Random(handle, A.cols()));
    Vector<double> b(A*true_x);

    return LinSysSolnPair(TypedLinearSystem<M, double>(A, b), true_x);

}

void create_or_clear_directory(fs::path dir, Experiment_Log logger);

template <template <template <typename> typename> typename Solver, template <typename> typename M>
Experiment_Data<Solver, M> execute_solve(
    std::shared_ptr<Solver<M>> arg_solver_ptr,
    bool show_plots
) {

    Experiment_Clock exp_clock;
    exp_clock.start_clock_experiment();
    arg_solver_ptr->solve();
    if (show_plots) { arg_solver_ptr->view_relres_plot("log"); }
    exp_clock.stop_clock_experiment();

    return Experiment_Data<Solver, M>(exp_clock, arg_solver_ptr);

}

template <template <template <typename> typename> typename Solver, template <typename> typename M>
void run_record_FPGMRES_solve(
    std::shared_ptr<Solver<M>> arg_solver_ptr,
    std::string matrix_name,
    std::string solve_name,
    int exp_iter,
    fs::path output_dir,
    bool show_plots,
    Experiment_Log logger
) {
    std::string solve_experiment_id = std::format("{}_{}_{}", matrix_name, solve_name, exp_iter);
    logger.info(std::format("Running solve experiment: {}", solve_experiment_id));
    Experiment_Data<GenericIterativeSolve, M> data = execute_solve<GenericIterativeSolve, M>(
        arg_solver_ptr,
        show_plots
    );
    logger.info(data.get_info_string());
    record_FPGMRES_experimental_data_json(data, solve_experiment_id, output_dir, logger);
}

template <template <typename> typename M>
void run_record_MPGMRES_solve(
    std::shared_ptr<MP_GMRES_IR_Solve<M>> arg_solver_ptr,
    std::string matrix_name,
    std::string solve_name,
    int exp_iter,
    fs::path output_dir,
    bool show_plots,
    Experiment_Log logger
) {
    std::string solve_experiment_id = std::format("{}_{}_{}", matrix_name, solve_name, exp_iter);
    logger.info(std::format("Running solve experiment: {}", solve_experiment_id));
    Experiment_Data<MP_GMRES_IR_Solve, M> data = execute_solve<MP_GMRES_IR_Solve, M>(
        arg_solver_ptr,
        show_plots
    );
    logger.info(data.get_info_string());
    record_MPGMRES_experimental_data_json(data, solve_experiment_id, output_dir, logger);
}

template <template <typename> typename M>
void run_solve_group(
    cublasHandle_t handle,
    Solve_Group solve_group,
    fs::path data_dir,
    fs::path output_dir,
    Experiment_Log outer_logger
) {

    outer_logger.info("Running solve group: "+solve_group.id);

    fs::path solve_group_dir = output_dir / fs::path(solve_group.id);
    create_or_clear_directory(solve_group_dir, outer_logger);

    Experiment_Log logger(
        solve_group.id + "_logger", solve_group_dir / fs::path(solve_group.id + ".log"), false
    );
    logger.info(std::format("Solve info: {}", solve_group.solver_args.get_info_string()));

    bool show_plots = false;

    for (std::string matrix_name : solve_group.matrices_to_test) {
        for (int exp_iter = 0; exp_iter < solve_group.experiment_iterations; ++exp_iter) {

            LinSysSolnPair<M> lin_sys_pair = load_linear_problem<M>(handle, data_dir, matrix_name, logger);
            TypedLinearSystem<M, double> lin_sys_dbl(lin_sys_pair.first);

            if (solve_group.solver_suite_type == "all") {

                TypedLinearSystem<M, __half> lin_sys_hlf(lin_sys_dbl.get_A(), lin_sys_dbl.get_b());

                run_record_FPGMRES_solve<GenericIterativeSolve, M>(
                    std::make_shared<FP_GMRES_IR_Solve<M, __half>>(
                        lin_sys_hlf, u_hlf, solve_group.solver_args
                    ),
                    matrix_name, "FPGMRES16", exp_iter,
                    solve_group_dir,
                    false, logger
                );

                TypedLinearSystem<M, float> lin_sys_sgl(lin_sys_dbl.get_A(), lin_sys_dbl.get_b());

                run_record_FPGMRES_solve<GenericIterativeSolve, M>(
                    std::make_shared<FP_GMRES_IR_Solve<M, float>>(
                        lin_sys_sgl, u_sgl, solve_group.solver_args
                    ),
                    matrix_name, "FPGMRES32", exp_iter,
                    solve_group_dir,
                    false, logger
                );

            }

            if ((solve_group.solver_suite_type == "all") || (solve_group.solver_suite_type == "FP64_MP")) {

                run_record_FPGMRES_solve<GenericIterativeSolve, M>(
                    std::make_shared<FP_GMRES_IR_Solve<M, double>>(
                        lin_sys_dbl, u_dbl, solve_group.solver_args
                    ),
                    matrix_name, "FPGMRES64", exp_iter,
                    solve_group_dir,
                    false, logger
                );

                run_record_MPGMRES_solve<M>(
                    std::make_shared<SimpleConstantThreshold<M>>(
                        lin_sys_dbl, solve_group.solver_args
                    ),
                    matrix_name, "MPGMRES", exp_iter,
                    solve_group_dir,
                    false, logger
                );

            }

        }
    }

}

void run_experimental_spec(
    cublasHandle_t handle,
    Experiment_Specification exp_spec,
    fs::path data_dir,
    fs::path output_dir,
    Experiment_Log logger
);

#endif