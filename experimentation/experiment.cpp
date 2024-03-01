#include <memory>

#include <filesystem>
#include <iostream>
#include <string>
#include <sstream>

#include <cmath>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "experiment.h"
#include "experiment_tools.h"

#include "tools/read_matrix.h"
#include "types/types.h"

#include "solvers/IterativeSolve.h"
#include "solvers/nested/GMRES_IR/MP_GMRES_IR.h"
#include "solvers/nested/GMRES_IR/FP_GMRES_IR.h"

namespace fs = std::filesystem;

const double u_hlf = std::pow(2, -10);
const double u_sgl = std::pow(2, -23);
const double u_dbl = std::pow(2, -52);

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

int main() {

    std::cout << "*** Start Numerical Experimentation: experiment.cpp ***\n" << std::endl;

    fs::path input_dir_path("C:\\Users\\dosre\\dev\\numerical_experimentation\\input");
    std::cout << "Input directory: " << input_dir_path << std::endl;

    fs::path output_dir_path("C:\\Users\\dosre\\dev\\numerical_experimentation\\output");
    std::cout << "Output directory: " << output_dir_path << std::endl;

    std::ifstream csv_load_order;
    fs::path csv_load_order_path(input_dir_path / fs::path("csv_load_order.txt"));
    csv_load_order.open(csv_load_order_path);
    std::cout << "csv load order file: " << csv_load_order_path << std::endl << std::endl;

    if (!csv_load_order.is_open()) {
        throw std::runtime_error("csv_load_order did not load correctly");
    }

    fs::path matrix_path(input_dir_path / fs::path("experiment_matrices"));

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

    std::string temp_str;
    std::getline(csv_load_order, temp_str);

    std::cout << "Loading: " << matrix_path / fs::path(temp_str) << std::endl;
    MatrixDense<double> A(
        read_matrixCSV<MatrixDense, double>(
            handle,
            matrix_path / fs::path(temp_str)
        )
    );
    A.normalize_magnitude();
    std::cout << A.get_info_string() << std::endl;

    Vector<double> true_x(Vector<double>::Random(handle, A.cols()));
    Vector<double> b(A*true_x);

    SolveArgPkg solve_args;
    solve_args.init_guess = Vector<double>::Zero(handle, A.cols());
    solve_args.max_iter = 200;
    solve_args.max_inner_iter = 50;
    // solve_args.max_iter = 10;
    // solve_args.max_inner_iter = 10;
    solve_args.target_rel_res = std::pow(10, -10);

    TypedLinearSystem<MatrixDense, double> lin_sys_dbl(A, b);
    TypedLinearSystem<MatrixDense, float> lin_sys_sgl(A, b);
    TypedLinearSystem<MatrixDense, __half> lin_sys_hlf(A, b);

    bool show_plots = false;

    std::cout << "\nStarting FPGMRES64" << std::endl;
    std::cout << solve_args.get_info_string() << std::endl;
    Experiment_Data<GenericIterativeSolve, MatrixDense> fpgmres64_data = run_solve_experiment<GenericIterativeSolve, MatrixDense>(
        std::make_shared<FP_GMRES_IR_Solve<MatrixDense, double>>(lin_sys_dbl, u_dbl, solve_args),
        show_plots
    );
    std::cout << fpgmres64_data.get_info_string() << std::endl;
    record_experimental_data_json(fpgmres64_data, "FPGMRES64", output_dir_path);

    std::cout << "\nStarting FPGMRES32" << std::endl;
    std::cout << solve_args.get_info_string() << std::endl;
    Experiment_Data<GenericIterativeSolve, MatrixDense> fpgmres32_data = run_solve_experiment<GenericIterativeSolve, MatrixDense>(
        std::make_shared<FP_GMRES_IR_Solve<MatrixDense, float>>(lin_sys_sgl, u_sgl, solve_args),
        show_plots
    );
    std::cout << fpgmres32_data.get_info_string() << std::endl;
    record_experimental_data_json(fpgmres32_data, "FPGMRES32", output_dir_path);

    std::cout << "\nStarting FPGMRES16" << std::endl;
    std::cout << solve_args.get_info_string() << std::endl;
    Experiment_Data<GenericIterativeSolve, MatrixDense> fpgmres16_data = run_solve_experiment<GenericIterativeSolve, MatrixDense>(
        std::make_shared<FP_GMRES_IR_Solve<MatrixDense, __half>>(lin_sys_hlf, u_hlf, solve_args),
        show_plots
    );
    std::cout << fpgmres16_data.get_info_string() << std::endl;
    record_experimental_data_json(fpgmres16_data, "FPGMRES16", output_dir_path);

    std::cout << "\nStarting MPGMRES" << std::endl;
    std::cout << solve_args.get_info_string() << std::endl;
    Experiment_Data<MP_GMRES_IR_Solve, MatrixDense> mpgmres_data = run_solve_experiment<MP_GMRES_IR_Solve, MatrixDense>(
        std::make_shared<SimpleConstantThreshold<MatrixDense>>(lin_sys_dbl, solve_args),
        show_plots
    );
    std::cout << mpgmres_data.get_info_string() << std::endl;
    record_MPGMRES_experimental_data_json(mpgmres_data, "MPGMRES", output_dir_path);

    std::cout << "\n*** Finish Numerical Experimentation ***" << std::endl;
    
    return 0;

}