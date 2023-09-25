#ifndef SOLVERECORDER_H
#define SOLVERECORDER_H

#include <filesystem>

#include "tools/MatrixWriter.h"
#include "solvers/IterativeSolve.h"

using std::filesystem::create_directories;

template <template <typename> typename M>
void record_solve(
    const shared_ptr<GenericIterativeSolve<M>> &solver,
    const string save_dir
) {

    create_directories(save_dir);

    MatrixXd res_hist = solver->get_res_hist();
    string f_res_hist_path = save_dir + "res_hist.csv";
    write_matrixCSV(res_hist, f_res_hist_path);

    vector<double> res_norm_hist = solver->get_res_norm_hist();
    string f_res_norm_hist_path = save_dir + "res_norm_hist.csv";
    MatrixXd red_norm_hist_mat(res_norm_hist.size(), 1);
    for (int i=0; i<res_norm_hist.size(); ++i) { red_norm_hist_mat(i) = res_norm_hist[i]; }
    write_matrixCSV(red_norm_hist_mat, f_res_norm_hist_path);

    MatrixVector<double> soln = solver->get_generic_soln();
    string f_soln_path = save_dir + "soln.csv";
    write_matrixCSV(static_cast<MatrixXd>(soln), f_soln_path);

}

#endif