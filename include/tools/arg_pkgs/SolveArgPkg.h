#ifndef SOLVEARGPKG_H
#define SOLVEARGPKG_H

#include "../../types/types.h"

#include <string>

namespace cascade {

struct SolveArgPkg
{
public:

    constexpr static int default_max_iter = -1;
    constexpr static int default_max_inner_iter = -1;
    constexpr static double default_target_rel_res = -1;
    inline const static Vector<double> default_init_guess = (
        Vector<double>::Zero(cuHandleBundle(), 0)
    );

    int max_iter;
    int max_inner_iter;
    double target_rel_res;
    Vector<double> init_guess;

    bool check_default_max_iter() const {
        return max_iter == default_max_iter;
    }
    bool check_default_max_inner_iter() const {
        return max_iter == default_max_inner_iter;
    }
    bool check_default_target_rel_res() const {
        return target_rel_res == default_target_rel_res;
    }
    bool check_default_init_guess() const {
        if (init_guess.rows() != default_init_guess.rows()) {
            return false;
        } else if (init_guess.cols() != default_init_guess.cols()) {
            return false;
        } else {
            return init_guess == default_init_guess;
        }
    }

    SolveArgPkg(
        int arg_max_iter = default_max_iter,
        int arg_max_inner_iter = default_max_inner_iter,
        double arg_target_rel_res = default_target_rel_res,
        Vector<double> arg_init_guess = default_init_guess
    ):
        max_iter(arg_max_iter),
        max_inner_iter(arg_max_inner_iter),
        target_rel_res(arg_target_rel_res),
        init_guess(arg_init_guess)
    {};

    std::string get_info_string() const {
        return(
            "Max iter: " + std::to_string(max_iter) + " | "
            "Max inner iter: " + std::to_string(max_inner_iter) + " | "
            "Target rel-res: " + std::to_string(target_rel_res)
        );
    }

};

}

#endif