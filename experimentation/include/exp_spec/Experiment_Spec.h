#ifndef EXPERIMENT_SPEC_H
#define EXPERIMENT_SPEC_H

#include "Solve_Group.h"

#include <vector>
#include <string>

struct Experiment_Spec
{
public:
    
    const std::string id;
    std::vector<Solve_Group> solve_groups;

    Experiment_Spec(std::string arg_id);

    void add_solve_group(Solve_Group solve_group);

};

#endif