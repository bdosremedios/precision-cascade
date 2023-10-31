#ifndef SOLVERECORDER_H
#define SOLVERECORDER_H

#include "solvers/IterativeSolve.h"

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

using std::ofstream;

void write_json_array_to_ofstream(
    const MatrixXd &mat,
    ofstream &f_out,
    const string &padding 
) {

    f_out << "[\n";
    for (int i=0; i<mat.rows()-1; ++i) {
        f_out << padding << padding << "[";
        for (int j=0; j<mat.cols()-1; ++j) { f_out << mat(i, j) << ","; }
        f_out << mat(i, mat.cols()-1);
        f_out << "],\n";
    }
    f_out << padding << padding << "[";
    for (int j=0; j<mat.cols()-1; ++j) { f_out << mat(mat.rows()-1, j) << ","; }
    f_out << mat(mat.rows()-1, mat.cols()-1);
    f_out << "]\n" << padding << "]";


} // end write_json_array_to_ofstream

template <template <typename> typename M>
void record_solve(
    const shared_ptr<GenericIterativeSolve<M>> &solver,
    const fs::path save_path,
    const string ID_name
) {

    ofstream file_out;
    file_out.open(save_path);
    file_out << std::scientific;
    file_out.precision(16);

    if (file_out.is_open()) {

        file_out << "{\n\t\"solver_name\" : \"" << typeid(*solver).name() << "\",\n";

        file_out << "\t\"ID\" : \"" << ID_name << "\",\n";

        file_out << "\t\"res_hist\" : ";
        // MatrixXd res_hist = solver->get_res_hist();
        // write_json_array_to_ofstream(res_hist, file_out, "\t");
        file_out << "[],\n";
        // file_out << ",\n";

        file_out << "\t\"res_norm_hist\" : ";
        vector<double> res_norm_hist = solver->get_res_norm_hist();
        MatrixXd res_norm_hist_mat(res_norm_hist.size(), 1);
        for (int i=0; i<res_norm_hist.size(); ++i) { res_norm_hist_mat(i) = res_norm_hist[i]; }
        write_json_array_to_ofstream(res_norm_hist_mat, file_out, "\t");
        file_out << ",\n";

        file_out << "\t\"soln\" : ";
        MatrixVector<double> soln = solver->get_generic_soln();
        write_json_array_to_ofstream(static_cast<MatrixXd>(soln), file_out, "\t");
        file_out << "\n";

        file_out << "}";
        file_out.close();

    } else { throw runtime_error("Failed to open for write: " + save_path.string()); }

}

#endif