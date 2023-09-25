#ifndef TEST_MP_GMRES_IR_COMPONENT_H
#define TEST_MP_GMRES_IR_COMPONENT_H

#include "../../test.h"

#include "solvers/nested/GMRES_IR/MP_GMRES_IR.h"

template <template <typename> typename M>
class MP_GMRES_IR_Solve_TestingMock: public MP_GMRES_IR_Solve<M>
{
public:

    using MP_GMRES_IR_Solve<M>::cascade_phase;
    using MP_GMRES_IR_Solve<M>::INIT_PHASE;
    using MP_GMRES_IR_Solve<M>::HLF_PHASE;
    using MP_GMRES_IR_Solve<M>::SGL_PHASE;
    using MP_GMRES_IR_Solve<M>::DBL_PHASE;

    using MP_GMRES_IR_Solve<M>::MP_GMRES_IR_Solve;

    int specified_phase;

    void determine_phase() { cascade_phase = specified_phase; }

};

#endif