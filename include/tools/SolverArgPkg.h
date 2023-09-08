#ifndef SOLVERARGPKG
#define SOLVERARGPKG

#include "Eigen/Dense"

using Eigen::Matrix, Eigen::Dynamic;

class SolverArgPkg {

    public:

        const int default_max_iter(-1);
        const int default_max_inner_iter(-1);
        const double default_target_rel_res(-1);
        const double default_basis_zero_tol(-1);
        const Matrix<double, Dynamic, 1> default_init_guess(Matrix<double, Dynamic, 1>::Zero(0, 1));

        int max_iter = default_max_iter;
        int max_inner_iter = default_max_inner_iter;
        double target_rel_res = default_target_rel_res;
        double basis_zero_tol = default_basis_zero_tol;
        Matrix<double, Dynamic, 1> init_guess = default_init_guess;

        bool check_default_max_iter() { return max_iter == default_max_iter; }
        bool check_default_max_inner_iter() { return max_iter == default_max_inner_iter; }
        bool check_default_target_rel_res() { return target_rel_res == default_target_rel_res; }
        bool check_default_basis_zero_tol() { return basis_zero_tol == default_basis_zero_tol; }
        bool check_default_init_guess() { return init_guess == default_init_guess; }

};

#endif