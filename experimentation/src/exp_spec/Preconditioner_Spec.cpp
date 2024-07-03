#include "exp_spec/Preconditioner_Spec.h"

const std::unordered_set<std::string> Preconditioner_Spec::valid_precond_ids {
    "none", "jacobi", "ilu0", "ilutp"
};