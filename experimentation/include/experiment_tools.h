#ifndef EXPERIMENT_TOOLS_H
#define EXPERIMENT_TOOLS_H

#include "tools/arg_pkgs/SolveArgPkg.h"

#include <filesystem>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_set>

namespace fs = std::filesystem;

using namespace cascade;

void check_dir_exists(fs::path dir);

struct Solve_Group_Precond_Specs {

    std::string name = "";
    double ilutp_tau = -1.0;
    int ilutp_p = -1;

    Solve_Group_Precond_Specs() {}

    Solve_Group_Precond_Specs(std::string arg_name): name(arg_name) {}

    Solve_Group_Precond_Specs(
        std::string arg_name, double arg_ilutp_tau, int arg_ilutp_p
    ):
        name(arg_name), ilutp_tau(arg_ilutp_tau), ilutp_p(arg_ilutp_p)
    {}

    bool is_default() const {
        return ((name == "") && (ilutp_tau == -1.0) && (ilutp_p == -1));
    }

    std::string get_spec_string() const {
        if ((ilutp_tau == -1.0) && (ilutp_p == -1)) {
            return name + "_NA_NA";
        } else {
            std::stringstream ilutp_tau_strm;
            ilutp_tau_strm << std::setprecision(3);
            ilutp_tau_strm << ilutp_tau;
            std::string ilutp_tau_str = ilutp_tau_strm.str();
            for (int i=0; i<ilutp_tau_str.size(); ++i) {
                if (ilutp_tau_str[i] == '.') {
                    ilutp_tau_str.erase(i, 1);
                    --i;
                }
            }
            return name + "_" + ilutp_tau_str + "_" + std::to_string(ilutp_p);
        }
    }

    Solve_Group_Precond_Specs(const Solve_Group_Precond_Specs &other) {
        *this = other;
    }

    Solve_Group_Precond_Specs &operator=(
        const Solve_Group_Precond_Specs &other
    ) {

        name = other.name;
        ilutp_tau = other.ilutp_tau;
        ilutp_p = other.ilutp_p;

        return *this;

    }

    bool operator==(const Solve_Group_Precond_Specs &other) const {
        return (
            (name == other.name) &&
            (ilutp_tau == other.ilutp_tau) &&
            (ilutp_p == other.ilutp_p)
        );
    }

};

struct Solve_Group {

    const std::string id;
    const int experiment_iterations;
    const std::vector<std::string> solvers_to_use;
    const std::string matrix_type;
    const SolveArgPkg solver_args;
    const Solve_Group_Precond_Specs precond_specs;
    const std::vector<std::string> matrices_to_test;

    static const std::unordered_set<std::string> valid_solvers;

    Solve_Group(
        std::string arg_id,
        std::vector<std::string> arg_solvers_to_use,
        std::string arg_matrix_type,
        int arg_experiment_iterations,
        int arg_solver_max_outer_iterations,
        int arg_solver_max_inner_iterations,
        double arg_solver_target_relres,
        Solve_Group_Precond_Specs arg_precond_specs,
        std::vector<std::string> arg_matrices_to_test
    );

};

struct Experiment_Specification
{
public:
    
    const std::string id;
    std::vector<Solve_Group> solve_groups;

    Experiment_Specification(std::string arg_id);

    void add_solve_group(Solve_Group solve_group);

};

class Experiment_Clock 
{
public:

    std::chrono::steady_clock clock;
    std::chrono::time_point<std::chrono::steady_clock> start;
    std::chrono::time_point<std::chrono::steady_clock> stop;
    std::chrono::milliseconds time_ms;
    bool clock_ticking = false;

    void start_clock_experiment();
    
    void stop_clock_experiment();

    int get_elapsed_time_ms() const;

    std::string get_info_string() const;

};

template <
    template <template <typename> typename> typename TSolver,
    template <typename> typename TMatrix
>
struct Experiment_Data
{
public:
    
    Experiment_Clock clock;
    std::shared_ptr<TSolver<TMatrix>> solver_ptr;

    Experiment_Data(
        Experiment_Clock arg_clock,
        std::shared_ptr<TSolver<TMatrix>> arg_solver_ptr
    ):
        clock(arg_clock), solver_ptr(arg_solver_ptr) 
    {}

    Experiment_Data(const Experiment_Data &other) = default;
    Experiment_Data & operator=(const Experiment_Data &other) = default;

    std::string get_info_string() const {
        return clock.get_info_string() + " | " + solver_ptr->get_info_string();
    }

};

#endif