#ifndef SOLVEARGPKG_H
#define SOLVEARGPKG_H

#include "../types/types.h"

#include <memory>

using std::make_shared, std::shared_ptr;

class SolveArgPkg
{
public:

    constexpr static int default_max_iter = -1;
    constexpr static int default_max_inner_iter = -1;
    constexpr static double default_target_rel_res = -1;
    inline const static MatrixVector<double> default_init_guess = MatrixVector<double>::Zero(0);

    int max_iter;
    int max_inner_iter;
    double target_rel_res;
    MatrixVector<double> init_guess;

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

    // *** CONSTRUCTORS ***

    SolveArgPkg(
        int arg_max_iter = default_max_iter,
        int arg_max_inner_iter = default_max_inner_iter,
        double arg_target_rel_res = default_target_rel_res,
        MatrixVector<double> arg_init_guess = default_init_guess
    ):
        max_iter(arg_max_iter),
        max_inner_iter(arg_max_inner_iter),
        target_rel_res(arg_target_rel_res),
        init_guess(arg_init_guess)
    {};

};

#endif