#ifndef TEST_MP_GMRES_IR_COMPONENT_H
#define TEST_MP_GMRES_IR_COMPONENT_H

#include "test.h"

#include "solvers/nested/GMRES_IR/MP_GMRES_IR.h"

template <template <typename> typename TMatrix>
class MP_GMRES_IR_Solve_TestingMock: public MP_GMRES_IR_Solve<TMatrix>
{
public:

    using MP_GMRES_IR_Solve<TMatrix>::cascade_phase;
    using MP_GMRES_IR_Solve<TMatrix>::INIT_PHASE;
    using MP_GMRES_IR_Solve<TMatrix>::HLF_PHASE;
    using MP_GMRES_IR_Solve<TMatrix>::SGL_PHASE;
    using MP_GMRES_IR_Solve<TMatrix>::DBL_PHASE;

    using MP_GMRES_IR_Solve<TMatrix>::inner_solver;
    using MP_GMRES_IR_Solve<TMatrix>::outer_iterate_setup;

    using MP_GMRES_IR_Solve<TMatrix>::MP_GMRES_IR_Solve;

    int set_phase_to_use;

    int determine_next_phase() override { return set_phase_to_use; }

};

#endif