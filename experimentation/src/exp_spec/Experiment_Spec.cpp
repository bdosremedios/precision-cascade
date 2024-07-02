#include "exp_spec/Experiment_Spec.h"

Experiment_Spec::Experiment_Spec(std::string arg_id):
    id(arg_id)
{}

void Experiment_Spec::add_solve_group(Solve_Group solve_group) {
    solve_groups.push_back(solve_group);
}