#ifndef SOLVERECORDER_H
#define SOLVERECORDER_H

#include "solvers/IterativeSolve.h"

template <template <typename> typename M>
class SolveRecorder
{
private:

public:

    SolveRecorder(
        const shared_ptr<GenericIterativeSolve<M>> &solver,
        const string save_dir
    ) {
        
        MatrixXd res_hist = solver.get_res_hist();
        string f_res_hist_path = save_dir + "res_hist.csv";

        vector<double> res_norm_hist = solver.get_res_norm_hist();
        string f_res_norm_hist_path = save_dir + "res_norm_hist.csv";

        MatrixVector<double> soln = solver.get_generic_soln();
        string f_soln_path = save_dir + "soln.csv";



    }

};

#endif