#include "exp_read/exp_read.h"

#include <fstream>

int extract_integer(json::iterator member) {

    if (member->is_number_integer()) {
        return *member;
    } else {
        throw std::runtime_error(
            "extract_solve_group: extract_integer invalid value for key "
            "\"" + member.key() + "\""
        );
    }

}

std::vector<std::string> extract_solvers_to_use(json::iterator member) {

    std::vector<std::string> solvers_to_use;
    if (member->is_array()) {
        for (json::iterator it = member->begin(); it != member->end(); ++it) {
            if (it->is_string()) {
                solvers_to_use.push_back(*it);
            } else {
                throw std::runtime_error(
                    "extract_solve_group: extract_solvers_to_use invalid "
                    "solver or repeat solver "
                    "\"" + member.key() + "\""
                );
            }
        }
        return solvers_to_use;
    } else {
        throw std::runtime_error(
            "extract_solve_group: extract_solvers_to_use invalid value for key "
            "\"" + member.key() + "\""
        );
    }

}

std::string extract_matrix_type(json::iterator member) {

    if (member->is_string()) {
        return *member;
    } else {
        throw std::runtime_error(
            "extract_solve_group: extract_matrix_type invalid value for key"
            "\"" + member.key() + "\""
        );
    }

}

Preconditioner_Spec extract_solve_group_precond_specs(
    json::iterator member
) {

    if (member->is_string()) {

        return Preconditioner_Spec(*member);

    } else if (member->is_array()) {

        json::iterator it = member->begin();
        std::string name;
        double ilutp_tau;
        double ilutp_p;
        if (it != member->end() && it->is_string() && (*it == "ilutp")) {
            name = *it;
        } else {
            throw std::runtime_error(
                "extract_solve_group: extract_solve_group_precond_specs "
                "invalid array in key precond_specs"
            );
        }
        ++it;
        if (it != member->end() && it->is_number_float()) {
            ilutp_tau = *it;
        } else {
            throw std::runtime_error(
                "extract_solve_group: extract_solve_group_precond_specs "
                "invalid array in key precond_specs"
            );
        }
        ++it;
        if (it != member->end() && it->is_number_integer()) {
            ilutp_p = *it;
        } else {
            throw std::runtime_error(
                "extract_solve_group: extract_solve_group_precond_specs "
                "invalid array in key precond_specs"
            );
        }
        if (++it != member->end()) {
            throw std::runtime_error(
                "extract_solve_group: extract_solve_group_precond_specs too "
                "many values in array"
            );
        }
        return Preconditioner_Spec(name, ilutp_tau, ilutp_p);

    } else {
        throw std::runtime_error(
            "extract_solve_group: extract_solve_group_precond_specs invalid "
            "value for key precond_specs"
        );
    }

}

double extract_double(json::iterator member) {

    if (member->is_number_float()) {
        return *member;
    } else {
        throw std::runtime_error(
            "extract_solve_group: extract_double invalid value for key "
            "\"" + member.key() + "\""
        );
    }

}

std::vector<std::string> extract_string_vector(json::iterator member) {

    std::vector<std::string> matrix_names;
    if (member->is_array()) {

        for (json::iterator it = member->begin(); it != member->end(); ++it) {
            if (it->is_string()) {
                matrix_names.push_back(*it);
            } else {
                throw std::runtime_error(
                    "extract_solve_group: extract_string_vector in id "
                    "\"" + member.key() + "\" non string member"
                );
            }
        }

        return matrix_names;

    } else {
        throw std::runtime_error(
            "extract_solve_group: extract_string_vector invalid value for key "
            "\"" + member.key() + "\""
        );
    }

}

Solve_Group check_and_extract_solve_group(std::string id, json cand_obj) {

    int member_count = 0;
    int experiment_iterations = -1;
    std::vector<std::string> solvers_to_use;
    std::string matrix_type = "";
    int solver_max_outer_iterations = -1;
    int solver_max_inner_iterations = -1;
    double solver_target_relres = -1.;
    Preconditioner_Spec precond_specs;
    std::vector<std::string> matrices_to_test;
    for (
        json::iterator it = cand_obj.begin();
        (it != cand_obj.end()) && (member_count < 8);
        ++it
    ) {

        if (it.key() == "experiment_iterations") {
            experiment_iterations = extract_integer(it);
        } else if (it.key() == "solvers_to_use") {
            solvers_to_use = extract_solvers_to_use(it);
        } else if (it.key() == "matrix_type") {
            matrix_type = extract_matrix_type(it);
        } else if (it.key() == "solver_max_outer_iterations") {
            solver_max_outer_iterations = extract_integer(it);
        } else if (it.key() == "solver_max_inner_iterations") {
            solver_max_inner_iterations = extract_integer(it);
        } else if (it.key() == "solver_target_relres") {
            solver_target_relres = extract_double(it);
        } else if (it.key() == "precond_specs") {
            precond_specs = extract_solve_group_precond_specs(it);
        } else if (it.key() == "matrices_to_test") {
            matrices_to_test = extract_string_vector(it);
        } else {
            throw std::runtime_error(
                "extract_solve_group: in solve group id "
                "\"" + id + "\" invalid key \"" + it.key() + "\""
            );
        }
        ++member_count;

    }

    if (
        (member_count != 8) ||
        (experiment_iterations == -1) ||
        (solvers_to_use.size() == 0) ||
        (matrix_type == "") ||
        (solver_max_outer_iterations == -1) ||
        (solver_max_inner_iterations == -1) ||
        (solver_target_relres == -1.) ||
        (precond_specs.is_default()) ||
        (matrices_to_test.size() == 0)
    ) {
        throw std::runtime_error(
            "extract_solve_group: incorrect members for solve group id "
            "\"" + id + "\""
        );
    }

    return Solve_Group(
        id,
        solvers_to_use,
        matrix_type,
        experiment_iterations,
        solver_max_outer_iterations,
        solver_max_inner_iterations,
        solver_target_relres,
        precond_specs,
        matrices_to_test
    );

}

Experiment_Spec parse_experiment_spec(fs::path exp_spec_path) {

    std::ifstream exp_spec_stream(exp_spec_path);

    if (!exp_spec_stream.is_open()) {
        throw std::runtime_error(
            "parse_experiment_spec: failed to open " +
            exp_spec_path.string()
        );
    }
    
    std::string exp_spec_id = exp_spec_path.stem().string();
    Experiment_Spec exp_spec = Experiment_Spec(exp_spec_id);

    json exp_spec_json;
    try {
        exp_spec_json = json::parse(exp_spec_stream);
    } catch (json::parse_error e) {
        throw std::runtime_error(
            "parse_experiment_spec: error in json::parse of experiment "
            "specification id \"" + exp_spec_id + "\" - " + e.what()
        );
    }

    for (
        json::iterator iter = exp_spec_json.begin();
        iter != exp_spec_json.end();
        ++iter
    ) {

        if (iter->is_object()) {

            Solve_Group solve_group = check_and_extract_solve_group(
                iter.key(), *iter
            );
            exp_spec.add_solve_group(solve_group);

        } else {

            throw std::runtime_error(
                "parse_experiment_spec: solve group with id \"" + iter.key() +
                "\" is not a json object"
            );

        }

    }

    return exp_spec;

}