#ifndef TEST_ITERATIVEREFINEMENT_H
#define TEST_ITERATIVEREFINEMENT_H

#include "test.h"

#include "solvers/nested/IterativeRefinement.h"

using namespace cascade;


template <template <typename> typename TMatrix>
class InnerSolver_Mock: public GenericIterativeSolve<TMatrix>
{
public:



};

#endif