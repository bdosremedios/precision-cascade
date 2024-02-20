#ifndef SOLVERECORDER_H
#define SOLVERECORDER_H

#include <filesystem>
#include <fstream>
#include <string>
#include <memory>

#include "solvers/IterativeSolve.h"

namespace fs = std::filesystem;

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

class Experiment {};

#endif