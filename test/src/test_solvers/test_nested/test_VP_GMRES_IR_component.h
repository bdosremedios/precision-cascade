#ifndef TEST_VP_GMRES_IR_COMPONENT_H
#define TEST_VP_GMRES_IR_COMPONENT_H

#include "test.h"

#include "solvers/nested/GMRES_IR/VP_GMRES_IR.h"

template <template <typename> typename TMatrix>
class VP_GMRES_IR_Solve_TestingMock: public VP_GMRES_IR_Solve<TMatrix>
{
public:

    using VP_GMRES_IR_Solve<TMatrix>::cascade_phase;
    using VP_GMRES_IR_Solve<TMatrix>::INIT_PHASE;
    using VP_GMRES_IR_Solve<TMatrix>::HLF_PHASE;
    using VP_GMRES_IR_Solve<TMatrix>::SGL_PHASE;
    using VP_GMRES_IR_Solve<TMatrix>::DBL_PHASE;

    using VP_GMRES_IR_Solve<TMatrix>::inner_solver;
    using VP_GMRES_IR_Solve<TMatrix>::outer_iterate_setup;

    using VP_GMRES_IR_Solve<TMatrix>::VP_GMRES_IR_Solve;

    int set_phase_to_use;

    int determine_next_phase() override { return set_phase_to_use; }

};

#endif