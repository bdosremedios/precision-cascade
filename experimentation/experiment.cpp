#include <filesystem>
#include <memory>
#include <cmath>
#include <string>
#include <iostream>
#include <sstream>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "solvers/nested/GMRES_IR/MP_GMRES_IR.h"
#include "solvers/nested/GMRES_IR/FP_GMRES_IR.h"

// #include "types/types.h"
#include "tools/tools.h"

namespace fs = std::filesystem;

const double u_hlf = std::pow(2, -10);
const double u_sgl = std::pow(2, -23);
const double u_dbl = std::pow(2, -52);

// string get_file_name(fs::path file_path) {

//     string temp = file_path.string();
//     temp = temp.substr(temp.find_last_of("//")+1);
//     temp = temp.substr(0, temp.find_last_of("."));

//     return temp;

// }

// template <template <typename> typename M>
// void print_solver_info(
//     shared_ptr<GenericIterativeSolve<M>> solver,
//     string ID
// ) {
//     std::cout << "Name: " << ID << " | ";
//     std::cout << "Converged: " << solver->check_converged() << " | ";
//     std::cout << "Iter: " << solver->get_iteration() << " | ";
//     std::cout << "Relres: " << solver->get_relres() << std::endl;
// }

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
    A.self_scale_magnitude();
    A.view_properties();

    Vector<double> true_x(Vector<double>::Random(handle, A.cols()));
    Vector<double> b(A*true_x);

    SolveArgPkg solve_args;
    solve_args.init_guess = Vector<double>::Zero(handle, A.cols());
    solve_args.max_iter = 200;
    solve_args.max_inner_iter = 50;
    solve_args.target_rel_res = std::pow(10, -10);

    TypedLinearSystem<MatrixDense, double> lin_sys_dbl(A, b);
    TypedLinearSystem<MatrixDense, float> lin_sys_sgl(A, b);
    TypedLinearSystem<MatrixDense, __half> lin_sys_hlf(A, b);

    std::cout << "Starting FPGMRES64" << std::endl;
    std::shared_ptr<GenericIterativeSolve<MatrixDense>> fpgmres_dbl = (
        std::make_shared<FP_GMRES_IR_Solve<MatrixDense, double>>(lin_sys_dbl, u_dbl, solve_args)
    );
    fpgmres_dbl->solve();
    fpgmres_dbl->view_relres_plot("log");

    std::cout << "Starting FPGMRES32" << std::endl;
    std::shared_ptr<GenericIterativeSolve<MatrixDense>> fpgmres_sgl = (
        std::make_shared<FP_GMRES_IR_Solve<MatrixDense, float>>(lin_sys_sgl, u_sgl, solve_args)
    );
    fpgmres_sgl->solve();
    fpgmres_sgl->view_relres_plot("log");

    std::cout << "Starting FPGMRES16" << std::endl;
    std::shared_ptr<GenericIterativeSolve<MatrixDense>> fpgmres_hlf = (
        std::make_shared<FP_GMRES_IR_Solve<MatrixDense, __half>>(lin_sys_hlf, u_hlf, solve_args)
    );
    fpgmres_hlf->solve();
    fpgmres_hlf->view_relres_plot("log");

    std::cout << "Starting MPGMRES" << std::endl;
    std::shared_ptr<GenericIterativeSolve<MatrixDense>> mpgmres = (
        std::make_shared<SimpleConstantThreshold<MatrixDense>>(lin_sys_dbl, solve_args)
    );
    mpgmres->solve();
    mpgmres->view_relres_plot("log");


//     fs::directory_iterator iter(load_dir);
//     fs::directory_iterator curr = fs::begin(iter);

//     for (fs::directory_iterator curr = fs::begin(iter); curr != fs::end(iter); ++curr) {

//         MatrixDense<double> A_dense = read_matrixCSV<MatrixDense, double>(*curr);
//         A_dense = 1/(A_dense.maxCoeff())*A_dense;
//         MatrixSparse<double> A = A_dense.sparseView();

//         std::cout << "Testing: " << *curr << " of size " << A.rows() << "x" << A.cols() << std::endl;

//         SolveArgPkg solve_args;
//         solve_args.init_guess = MatrixVector<double>::Zero(A.cols());
//         solve_args.max_iter = 50;
//         solve_args.max_inner_iter = static_cast<int>(0.2*A.rows());
//         solve_args.target_rel_res = pow(10, -10);

//         for (int i=1; i<=3; ++i) {

//             string ID_prefix = get_file_name(*curr) + "_" + to_string(i);
//             MatrixVector<double> b = A*MatrixVector<double>::Random(A.cols());

//             shared_ptr<GenericIterativeSolve<MatrixSparse>> fpgmres_hlf = (
//                 make_shared<FP_GMRES_IR_Solve<MatrixSparse, half>>(A, b, u_hlf, solve_args)
//             );
//             fpgmres_hlf->solve();
//             record_solve(fpgmres_hlf,
//                          save_dir / fs::path(ID_prefix+"_fphlf.json"),
//                          ID_prefix+"_fphlf");
//             print_solver_info(fpgmres_hlf, ID_prefix+"_fphlf");

//             shared_ptr<GenericIterativeSolve<MatrixSparse>> fpgmres_sgl = (
//                 make_shared<FP_GMRES_IR_Solve<MatrixSparse, float>>(A, b, u_sgl, solve_args)
//             );
//             fpgmres_sgl->solve();
//             record_solve(fpgmres_sgl,
//                          save_dir / fs::path(ID_prefix+"_fpsgl.json"),
//                          ID_prefix+"_fpsgl");
//             print_solver_info(fpgmres_sgl, ID_prefix+"_fpsgl");

//             shared_ptr<GenericIterativeSolve<MatrixSparse>> fpgmres_dbl = (
//                 make_shared<FP_GMRES_IR_Solve<MatrixSparse, double>>(A, b, u_dbl, solve_args)
//             );
//             fpgmres_dbl->solve();
//             record_solve(fpgmres_dbl,
//                          save_dir / fs::path(ID_prefix+"_fpdbl.json"),
//                          ID_prefix+"_fpdbl");
//             print_solver_info(fpgmres_dbl, ID_prefix+"_fpdbl");

//             shared_ptr<GenericIterativeSolve<MatrixSparse>> mpgmres = (
//                 make_shared<SimpleConstantThreshold<MatrixSparse>>(A, b, solve_args)
//             );
//             mpgmres->solve();
//             record_solve(mpgmres,
//                          save_dir / fs::path(ID_prefix+"_mp.json"),
//                          ID_prefix+"_mp");
//             print_solver_info(mpgmres, ID_prefix+"_mp");

//         }

//     }

    std::cout << "\n*** Finish Numerical Experimentation ***" << std::endl;
    
    return 0;

}