#ifndef SOLVERARGPKG
#define SOLVERARGPKG

#include "Eigen/Dense"

#include "../preconditioners/ImplementedPreconditioners.h"

#include <memory>

using Eigen::Matrix, Eigen::Dynamic;

using std::make_shared, std::shared_ptr;

class SolveArgPkg {

    public:

        const int default_max_iter = -1;
        const int default_max_inner_iter = -1;
        const double default_target_rel_res = -1;
        const Matrix<double, Dynamic, 1> default_init_guess = Matrix<double, Dynamic, 1>::Zero(0, 1);

        int max_iter = default_max_iter;
        int max_inner_iter = default_max_inner_iter;
        double target_rel_res = default_target_rel_res;
        Matrix<double, Dynamic, 1> init_guess = default_init_guess;

        bool check_default_max_iter() const { return max_iter == default_max_iter; }
        bool check_default_max_inner_iter() const { return max_iter == default_max_inner_iter; }
        bool check_default_target_rel_res() const { return target_rel_res == default_target_rel_res; }
        bool check_default_init_guess() const {
            if (init_guess.rows() != default_init_guess.rows()) {
                return false;
            } else if (init_guess.cols() != default_init_guess.cols()) {
                return false;
            } else {
                return init_guess == default_init_guess;
            }
        }

};

template <typename U>
class PrecondArgPkg {

    public:

        const shared_ptr<Preconditioner<U>> default_left_precond = make_shared<NoPreconditioner<U>>();
        const shared_ptr<Preconditioner<U>> default_right_precond = make_shared<NoPreconditioner<U>>();

        shared_ptr<Preconditioner<U>> left_precond = default_left_precond;
        shared_ptr<Preconditioner<U>> right_precond = default_right_precond;

};

#endif