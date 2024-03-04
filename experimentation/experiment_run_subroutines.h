#ifndef RUN_EXPERIMENT_H
#define RUN_EXPERIMENT_H

#include <filesystem>
#include <format>
#include <string>

#include "types/types.h"
#include "tools/read_matrix.h"

#include "experiment_record.h"

#include "solvers/IterativeSolve.h"
#include "solvers/nested/GMRES_IR/MP_GMRES_IR.h"
#include "solvers/nested/GMRES_IR/FP_GMRES_IR.h"

namespace fs = std::filesystem;

static const double u_hlf = std::pow(2, -10);
static const double u_sgl = std::pow(2, -23);
static const double u_dbl = std::pow(2, -52);

template <template <template <typename> typename> typename Solver, template <typename> typename M>
Experiment_Data<Solver, M> run_solve_experiment(
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

template <template <typename> typename M>
void run_FP64_MP_solves(
    const cublasHandle_t handle,
    const fs::path input_dir,
    const std::string matrix_file_name,
    const fs::path output_dir,
    const int max_iter,
    const int max_inner_iter,
    const double target_rel_res,
    const bool show_plots,
    const int iteration = 0
) {

    fs::path matrix_path = input_dir / fs::path(matrix_file_name+".csv");

    std::cout << std::format("Loading: {}", matrix_path.string()) << std::endl;

    M<double> A(read_matrixCSV<M, double>(handle, matrix_path));
    A.normalize_magnitude();
    std::cout << A.get_info_string() << std::endl;

    Vector<double> true_x(Vector<double>::Random(handle, A.cols()));

    Vector<double> b(A*true_x);
    std::cout << b.get_info_string() << std::endl;

    SolveArgPkg solve_args;
    solve_args.init_guess = Vector<double>::Zero(handle, A.cols());
    solve_args.max_iter = max_iter;
    solve_args.max_inner_iter = max_inner_iter;
    solve_args.target_rel_res = target_rel_res;

    TypedLinearSystem<M, double> lin_sys_dbl(A, b);
    TypedLinearSystem<M, float> lin_sys_sgl(A, b);
    TypedLinearSystem<M, __half> lin_sys_hlf(A, b);

    std::string fpgmres64_id = std::format("{}_FPGMRES64_{}", matrix_file_name, iteration);
    std::cout << std::format("\nStarting {}", fpgmres64_id) << std::endl;
    std::cout << solve_args.get_info_string() << std::endl;
    Experiment_Data<GenericIterativeSolve, M> fpgmres64_data = run_solve_experiment<GenericIterativeSolve, M>(
        std::make_shared<FP_GMRES_IR_Solve<M, double>>(lin_sys_dbl, u_dbl, solve_args),
        show_plots
    );
    std::cout << fpgmres64_data.get_info_string() << std::endl;
    record_experimental_data_json(fpgmres64_data, fpgmres64_id, output_dir);

    std::string mpgmres_id = std::format("{}_MPGMRES_{}", matrix_file_name, iteration);
    std::cout << std::format("\nStarting {}", mpgmres_id) << std::endl;
    std::cout << solve_args.get_info_string() << std::endl;
    Experiment_Data<MP_GMRES_IR_Solve, M> mpgmres_data = run_solve_experiment<MP_GMRES_IR_Solve, M>(
        std::make_shared<SimpleConstantThreshold<M>>(lin_sys_dbl, solve_args),
        show_plots
    );
    std::cout << mpgmres_data.get_info_string() << std::endl;
    record_MPGMRES_experimental_data_json(mpgmres_data, mpgmres_id, output_dir);

}

template <template <typename> typename M>
void run_all_solves(
    const cublasHandle_t handle,
    const fs::path input_dir,
    const std::string matrix_file_name,
    const fs::path output_dir,
    const int max_iter,
    const int max_inner_iter,
    const double target_rel_res,
    const bool show_plots,
    const int iteration = 0
) {

    fs::path matrix_path = input_dir / fs::path(matrix_file_name+".csv");

    std::cout << std::format("Loading: {}", matrix_path.string()) << std::endl;

    M<double> A(read_matrixCSV<M, double>(handle, matrix_path));
    A.normalize_magnitude();
    std::cout << A.get_info_string() << std::endl;

    Vector<double> true_x(Vector<double>::Random(handle, A.cols()));

    Vector<double> b(A*true_x);
    std::cout << b.get_info_string() << std::endl;

    SolveArgPkg solve_args;
    solve_args.init_guess = Vector<double>::Zero(handle, A.cols());
    solve_args.max_iter = max_iter;
    solve_args.max_inner_iter = max_inner_iter;
    solve_args.target_rel_res = target_rel_res;

    TypedLinearSystem<M, double> lin_sys_dbl(A, b);
    TypedLinearSystem<M, float> lin_sys_sgl(A, b);
    TypedLinearSystem<M, __half> lin_sys_hlf(A, b);

    std::string fpgmres64_id = std::format("{}_FPGMRES64_{}", matrix_file_name, iteration);
    std::cout << std::format("\nStarting {}", fpgmres64_id) << std::endl;
    std::cout << solve_args.get_info_string() << std::endl;
    Experiment_Data<GenericIterativeSolve, M> fpgmres64_data = run_solve_experiment<GenericIterativeSolve, M>(
        std::make_shared<FP_GMRES_IR_Solve<M, double>>(lin_sys_dbl, u_dbl, solve_args),
        show_plots
    );
    std::cout << fpgmres64_data.get_info_string() << std::endl;
    record_experimental_data_json(fpgmres64_data, fpgmres64_id, output_dir);

    std::string fpgmres32_id = std::format("{}_FPGMRES32_{}", matrix_file_name, iteration);
    std::cout << std::format("\nStarting {}", fpgmres32_id) << std::endl;
    std::cout << solve_args.get_info_string() << std::endl;
    Experiment_Data<GenericIterativeSolve, M> fpgmres32_data = run_solve_experiment<GenericIterativeSolve, M>(
        std::make_shared<FP_GMRES_IR_Solve<M, float>>(lin_sys_sgl, u_sgl, solve_args),
        show_plots
    );
    std::cout << fpgmres32_data.get_info_string() << std::endl;
    record_experimental_data_json(fpgmres32_data, fpgmres32_id, output_dir);

    std::string fpgmres16_id = std::format("{}_FPGMRES16_{}", matrix_file_name, iteration);
    std::cout << std::format("\nStarting {}", fpgmres16_id) << std::endl;
    std::cout << solve_args.get_info_string() << std::endl;
    Experiment_Data<GenericIterativeSolve, M> fpgmres16_data = run_solve_experiment<GenericIterativeSolve, M>(
        std::make_shared<FP_GMRES_IR_Solve<M, __half>>(lin_sys_hlf, u_hlf, solve_args),
        show_plots
    );
    std::cout << fpgmres16_data.get_info_string() << std::endl;
    record_experimental_data_json(fpgmres16_data, fpgmres16_id, output_dir);

    std::string mpgmres_id = std::format("{}_MPGMRES_{}", matrix_file_name, iteration);
    std::cout << std::format("\nStarting {}", mpgmres_id) << std::endl;
    std::cout << solve_args.get_info_string() << std::endl;
    Experiment_Data<MP_GMRES_IR_Solve, M> mpgmres_data = run_solve_experiment<MP_GMRES_IR_Solve, M>(
        std::make_shared<SimpleConstantThreshold<M>>(lin_sys_dbl, solve_args),
        show_plots
    );
    std::cout << mpgmres_data.get_info_string() << std::endl;
    record_MPGMRES_experimental_data_json(mpgmres_data, mpgmres_id, output_dir);

}

#endif