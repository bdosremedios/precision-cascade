#include "experiment_tools.h"

void check_dir_exists(fs::path dir) {
    if (!fs::exists(dir)) {
        throw std::runtime_error(
            "Directory " + dir.string() + " does not exist"
        );
    }
}

const std::unordered_set<std::string> Solve_Group::valid_solvers {
    "FP16", "FP32", "FP64", "SimpleConstantThreshold", "RestartCount"
};

Solve_Group::Solve_Group(
    std::string arg_id,
    std::vector<std::string> arg_solvers_to_use,
    std::string arg_matrix_type,
    int arg_experiment_iterations,
    int arg_solver_max_outer_iterations,
    int arg_solver_max_inner_iterations,
    double arg_solver_target_relres,
    Solve_Group_Precond_Specs arg_precond_specs,
    std::vector<std::string> arg_matrices_to_test
): 
    id(arg_id),
    solvers_to_use(arg_solvers_to_use),
    matrix_type(arg_matrix_type),
    experiment_iterations(arg_experiment_iterations),
    solver_args(
        arg_solver_max_outer_iterations,
        arg_solver_max_inner_iterations,
        arg_solver_target_relres
    ),
    precond_specs(arg_precond_specs),
    matrices_to_test(arg_matrices_to_test)
{

    // Check validity of solve_group parameters
    std::unordered_set<std::string> solvers_seen;
    for (std::string solver_id : solvers_to_use) {
        if (valid_solvers.count(solver_id) == 0) {
            throw std::runtime_error(
                "Solve_Group: invalid solver encountered in solvers_to_use "
                "\"" + solver_id + "\""
            );
        } else if (solvers_seen.count(solver_id) == 1) {
            throw std::runtime_error(
                "Solve_Group: repeated solver encountered in solvers_to_use "
                "\"" + solver_id + "\""
            );
        } else {
            solvers_seen.insert(solver_id);
        }
    }
    if (!((matrix_type == "dense") || (matrix_type == "sparse"))) {
        throw std::runtime_error(
            "Solve_Group: invalid matrix_type \"" + matrix_type + "\""
        );
    }
    if (experiment_iterations <= 0) {
        throw std::runtime_error(
            "Solve_Group: invalid experiment_iterations \"" +
            std::to_string(experiment_iterations) + "\""
        );
    }
    if (
        (arg_solver_max_outer_iterations < 0) ||
        (arg_solver_max_inner_iterations < 0) ||
        (arg_solver_target_relres < 0.)
    ) {
        throw std::runtime_error(
            "Solve_Group: invalid nested solver arguments"
        );
    }
    if (matrices_to_test.size() == 0) {
        throw std::runtime_error(
            "Solve_Group: empty matrices_to_test"
        );
    }
    for (std::string mat_file_name : matrices_to_test) {
        fs::path file_path(mat_file_name);
        if (
            !((file_path.extension() == ".mtx") ||
              (file_path.extension() == ".csv"))
        ) {
            throw std::runtime_error(
                "Solve_Group: invalid matrix in matrices_to_test \"" +
                mat_file_name + "\""
            );
        }
    }

}

Experiment_Specification::Experiment_Specification(std::string arg_id):
    id(arg_id)
{}

void Experiment_Specification::add_solve_group(Solve_Group solve_group) {
    solve_groups.push_back(solve_group);
}

void Experiment_Clock::start_clock_experiment() {
    if (!clock_ticking) {
        start = clock.now();
        clock_ticking = true;
    } else {
        throw std::runtime_error(
            "Experiment_Clock: start_clock_experiment clock already ticking"
        );
    } 
}
    
void Experiment_Clock::stop_clock_experiment() {
    if (clock_ticking) {
        stop = clock.now();
        clock_ticking = false;
        time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            stop-start
        );
    } else {
        throw std::runtime_error(
            "Experiment_Clock: stop_clock_experiment clock not ticking"
        );
    }
}

int Experiment_Clock::get_elapsed_time_ms() const {
    return time_ms.count();
}

std::string Experiment_Clock::get_info_string() const {
    return "Elapsed time (ms): " + std::to_string(get_elapsed_time_ms());
}

std::ofstream Timed_Experiment_Data::open_json_ofstream(
    std::string file_name, fs::path save_dir, Experiment_Log logger
) const {
    
    fs::path save_path(save_dir / fs::path(file_name + ".json"));
    logger.info("Save data to: " + save_path.string());
    
    std::ofstream file_out;
    file_out.open(save_path, std::ofstream::out);

    if (!file_out.is_open()) {
        throw std::runtime_error(
            "open_file_ofstream: Failed to open for write: " +
            save_path.string()
        );
    }

    return file_out;

}

void Timed_Experiment_Data::start_json(std::ofstream &file_out) const {
    file_out << "{\n";
}

void Timed_Experiment_Data::end_json(std::ofstream &file_out) const {
    file_out << "}";
    file_out.close();
}

// template <
//     template <template <typename> typename> typename TSolver,
//     template <typename> typename TMatrix
// >
// std::string Solve_Data<TSolver, TMatrix>::bool_to_string(bool b) const {
// }

// template <
//     template <template <typename> typename> typename TSolver,
//     template <typename> typename TMatrix
// >
// std::string Solve_Data<TSolver, TMatrix>::vector_to_jsonarray_str(
//     std::vector<double> vec, int padding_level
// ) const {

//     std::stringstream strm_to_write;
//     strm_to_write << std::setprecision(17);

//     for (int i=0; i<padding_level; ++i) {
//         strm_to_write << "\t";
//     }

//     strm_to_write << "[";
//     for (int i=0; i<vec.size()-1; ++i) {
//         strm_to_write << vec[i] << ", ";
//     }
//     strm_to_write << vec[vec.size()-1] << "]";

//     return strm_to_write.str();

// }