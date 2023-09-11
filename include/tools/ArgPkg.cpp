#include "ArgPkg.h"

const Matrix<double, 0, 1> SolveArgPkg::default_init_guess = Matrix<double, 0, 1>();

template<typename U>
const shared_ptr<Preconditioner<U>> PrecondArgPkg<U>::default_left_precond = make_shared<NoPreconditioner<U>>();
template<typename U>
const shared_ptr<Preconditioner<U>> PrecondArgPkg<U>::default_right_precond = make_shared<NoPreconditioner<U>>();