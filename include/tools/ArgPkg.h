#ifndef SOLVERARGPKG
#define SOLVERARGPKG

#include "Eigen/Dense"

#include "../preconditioners/ImplementedPreconditioners.h"

#include <memory>

using Eigen::Matrix, Eigen::Dynamic;

using std::make_shared, std::shared_ptr;

class SolveArgPkg {

    public:

        constexpr static int default_max_iter = -1;
        constexpr static int default_max_inner_iter = -1;
        constexpr static double default_target_rel_res = -1;
        const static Matrix<double, 0, 1> default_init_guess;

        int max_iter;
        int max_inner_iter;
        double target_rel_res;
        Matrix<double, Dynamic, 1> init_guess;

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

        // *** CONSTRUCTORS, CONSTRUCTOR OVERLOADS ***

        SolveArgPkg(
            int arg_max_iter = default_max_iter,
            int arg_max_inner_iter = default_max_inner_iter,
            double arg_target_rel_res = default_target_rel_res,
            Matrix<double, Dynamic, 1> arg_init_guess = default_init_guess
        ):
            max_iter(arg_max_iter),
            max_inner_iter(arg_max_inner_iter),
            target_rel_res(arg_target_rel_res),
            init_guess(arg_init_guess)
        {};

        void reset() {
            max_iter = default_max_iter;
            max_inner_iter = default_max_inner_iter;
            target_rel_res = default_target_rel_res;
            init_guess = default_init_guess;
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