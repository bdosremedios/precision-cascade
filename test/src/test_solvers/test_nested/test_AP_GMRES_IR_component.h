#ifndef TEST_AP_GMRES_IR_COMPONENT_H
#define TEST_AP_GMRES_IR_COMPONENT_H

#include "test.h"

#include "solvers/nested/GMRES_IR/AP_GMRES_IR.h"

template <template <typename> typename TMatrix>
class AP_GMRES_IR_Solve_TestingMock: public AP_GMRES_IR_Solve<TMatrix>
{
public:

    using AP_GMRES_IR_Solve<TMatrix>::cascade_phase;
    using AP_GMRES_IR_Solve<TMatrix>::INIT_PHASE;
    using AP_GMRES_IR_Solve<TMatrix>::HLF_PHASE;
    using AP_GMRES_IR_Solve<TMatrix>::SGL_PHASE;
    using AP_GMRES_IR_Solve<TMatrix>::DBL_PHASE;

    using AP_GMRES_IR_Solve<TMatrix>::inner_solver;
    using AP_GMRES_IR_Solve<TMatrix>::outer_iterate_setup;

    using AP_GMRES_IR_Solve<TMatrix>::AP_GMRES_IR_Solve;

    int set_phase_to_use;

    int determine_next_phase() override { return set_phase_to_use; }

};

#endif