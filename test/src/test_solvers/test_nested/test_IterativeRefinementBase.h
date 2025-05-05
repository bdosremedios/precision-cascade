#ifndef TEST_ITERATIVEREFINEMENTBASE_H
#define TEST_ITERATIVEREFINEMENTBASE_H

#include "test.h"

#include "solvers/IterativeSolve.h"
#include "solvers/nested/IterativeRefinementBase.h"

using namespace cascade;

template <template <typename> typename TMatrix>
class IterativeRefinementBaseSolver_Mock: public GenericIterativeSolve<TMatrix>
{
public:

    Vector<double> thing_to_quarter_add;

    IterativeRefinementBaseSolver_Mock(
        Vector<double> arg_thing_to_quarter_add,
        const GenericLinearSystem<TMatrix> * const arg_gen_lin_sys,
        const SolveArgPkg &arg_pkg

    ):
        thing_to_quarter_add(arg_thing_to_quarter_add),
        GenericIterativeSolve<TMatrix>(arg_gen_lin_sys, arg_pkg)
    {}

    void derived_generic_reset() override {}

    // Add half the solution
    void iterate() override {
        this->generic_soln += thing_to_quarter_add * Scalar<double>(0.25);
    }

    // Set thing_to_quarter_add to nan
    void set_thing_to_quarter_add_to_nan() {
        for (int i=0; i<thing_to_quarter_add.rows(); i++) {
            thing_to_quarter_add.set_elem(i, std::nan(""));
        }
    }

};

template <template <typename> typename TMatrix>
class IterativeRefinementBase_Mock: public IterativeRefinementBase<TMatrix>
{
public:

    bool init_inner_outer_hit = false;
    int outer_iterate_setup_hit_count = 0;

    Vector<double> soln_addition;

    using IterativeRefinementBase<TMatrix>::iterate;

    bool to_nan_next_iterate = false;

    void make_next_iterate_nan() {
        to_nan_next_iterate = true;
    }

    void initialize_inner_outer_solver() override {
        init_inner_outer_hit = true;
        std::shared_ptr<IterativeRefinementBaseSolver_Mock<TMatrix>> inner_mock = (
            std::make_shared<IterativeRefinementBaseSolver_Mock<TMatrix>>(
                soln_addition,
                this->gen_lin_sys_ptr,
                this->inner_solve_arg_pkg
            )
        );
        this->inner_solver = inner_mock;
        if (to_nan_next_iterate) {
            inner_mock->set_thing_to_quarter_add_to_nan();
        }
    }

    void outer_iterate_setup() override {
        outer_iterate_setup_hit_count += 1;
        std::shared_ptr<IterativeRefinementBaseSolver_Mock<TMatrix>> inner_mock = (
            std::make_shared<IterativeRefinementBaseSolver_Mock<TMatrix>>(
                soln_addition,
                this->gen_lin_sys_ptr,
                this->inner_solve_arg_pkg
            )
        );
        this->inner_solver = inner_mock;
        if (to_nan_next_iterate) {
            inner_mock->set_thing_to_quarter_add_to_nan();
        }
    }

    IterativeRefinementBase_Mock(
        Vector<double> arg_soln_addition,
        const GenericLinearSystem<TMatrix> * const arg_gen_lin_sys,
        const SolveArgPkg &arg_pkg
    ):
        soln_addition(arg_soln_addition),
        IterativeRefinementBase<TMatrix>(arg_gen_lin_sys, arg_pkg)
    {
        initialize_inner_outer_solver();
    }

};

#endif