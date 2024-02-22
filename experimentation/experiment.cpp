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

template <template <typename> typename M>
Experiment_Data<M> run_solve_experiment(
    std::shared_ptr<GenericIterativeSolve<M>> arg_solver_ptr,
    bool show_plots
) {

    Experiment_Clock exp_clock;
    exp_clock.start_clock_experiment();
    arg_solver_ptr->solve();
    if (show_plots) { arg_solver_ptr->view_relres_plot("log"); }
    exp_clock.stop_clock_experiment();

    return Experiment_Data<M>(exp_clock, arg_solver_ptr);

}

void write_mat_to_json_array_in_ofstream(
    const MatrixDense<double> &mat,
    std::ofstream &f_out,
    const std::string &padding 
) {

    double *h_mat = static_cast<double *>(malloc(mat.rows()*mat.cols()*sizeof(double)));

    mat.copy_data_to_ptr(h_mat, mat.rows(), mat.cols());

    f_out << "[\n";
    for (int i=0; i<mat.rows()-1; ++i) {
        f_out << padding << padding << "[";
        for (int j=0; j<mat.cols()-1; ++j) { f_out << h_mat[i+j*mat.rows()] << ","; }
        f_out << h_mat[i+(mat.cols()-1)*mat.rows()];
        f_out << "],\n";
    }
    f_out << padding << padding << "[";
    for (int j=0; j<mat.cols()-1; ++j) { f_out << h_mat[mat.rows()-1+j*mat.rows()] << ","; }
    f_out << h_mat[mat.rows()-1+(mat.cols()-1)*mat.rows()];
    f_out << "]\n" << padding << "]";

    free(h_mat);

}

void write_vec_to_json_array_in_ofstream(
    const Vector<double> &vec,
    std::ofstream &f_out,
    const std::string &padding 
) {

    double *h_vec = static_cast<double *>(malloc(vec.rows()*sizeof(double)));

    vec.copy_data_to_ptr(h_vec, vec.rows());

    f_out << "[\n";
    for (int i=0; i<vec.rows()-1; ++i) {
        f_out << padding << padding << "[" << h_vec[i] << "],\n";
    }
    f_out << padding << "]";

    free(h_vec);

}

template <template <typename> typename M>
void record_solve(
    const std::shared_ptr<GenericIterativeSolve<M>> &solver,
    const fs::path save_path,
    const std::string ID_name
) {

    std::ofstream file_out;
    file_out.open(save_path);
    file_out << std::scientific;
    file_out.precision(16);

    if (file_out.is_open()) {

        file_out << "{\n\t\"solver_name\" : \"" << typeid(*solver).name() << "\",\n";

        file_out << "\t\"ID\" : \"" << ID_name << "\",\n";

        // file_out << "\t\"res_hist\" : ";
        // // MatrixXd res_hist = solver->get_res_hist();
        // // write_matrix_to_json_array_in_ofstream(res_hist, file_out, "\t");
        // file_out << "[],\n";
        // // file_out << ",\n";

        file_out << "\t\"res_norm_hist\" : ";
        std::vector<double> res_norm_hist_vec = solver->get_res_norm_hist();
        Vector<double> res_norm_hist_mat(
            solver->get_generic_soln().get_handle(),
            res_norm_hist_vec.size()
        );
        for (int i=0; i<res_norm_hist_vec.size(); ++i) {
            res_norm_hist_mat.set_elem(i, Scalar<double>(res_norm_hist_vec[i]));
        }
        write_json_array_to_ofstream(res_norm_hist_mat, file_out, "\t");
        file_out << ",\n";

        file_out << "\t\"soln\" : ";
        Vector<double> soln(solver->get_generic_soln());
        write_json_array_to_ofstream(soln, file_out, "\t");
        file_out << "\n";

        file_out << "}";
        file_out.close();

    } else {

        throw runtime_error("Failed to open for write: " + save_path.string());

    }

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
    A.print_info();

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
    solve_args.print_info();
    Experiment_Data<MatrixDense> fpgmres64_data = run_solve_experiment<MatrixDense>(
        std::make_shared<FP_GMRES_IR_Solve<MatrixDense, double>>(lin_sys_dbl, u_dbl, solve_args),
        show_plots
    );
    std::cout << fpgmres64_data.get_info_string() << std::endl;

    std::cout << "\nStarting FPGMRES32" << std::endl;
    solve_args.print_info();
    Experiment_Data<MatrixDense> fpgmres32_data = run_solve_experiment<MatrixDense>(
        std::make_shared<FP_GMRES_IR_Solve<MatrixDense, float>>(lin_sys_sgl, u_sgl, solve_args),
        show_plots
    );
    std::cout << fpgmres32_data.get_info_string() << std::endl;

    std::cout << "\nStarting FPGMRES16" << std::endl;
    solve_args.print_info();
    Experiment_Data<MatrixDense> fpgmres16_data = run_solve_experiment<MatrixDense>(
        std::make_shared<FP_GMRES_IR_Solve<MatrixDense, __half>>(lin_sys_hlf, u_hlf, solve_args),
        show_plots
    );
    std::cout << fpgmres16_data.get_info_string() << std::endl;

    std::cout << "\nStarting MPGMRES" << std::endl;
    solve_args.print_info();
    Experiment_Data<MatrixDense> mpgmres_data = run_solve_experiment<MatrixDense>(
        std::make_shared<SimpleConstantThreshold<MatrixDense>>(lin_sys_dbl, solve_args),
        show_plots
    );
    std::cout << mpgmres_data.get_info_string() << std::endl;

    std::cout << "\n*** Finish Numerical Experimentation ***" << std::endl;
    
    return 0;

}