#ifndef EXPERIMENT_TOOLS_H
#define EXPERIMENT_TOOLS_H

#include <chrono>
#include <string>
#include <format>
#include <filesystem>
#include <iostream>
#include <fstream>

namespace fs = std::filesystem;

#include "solvers/IterativeSolve.h"

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

template <template <typename> typename M>
struct Experiment_Data
{
public:
    
    Experiment_Clock clock;
    std::shared_ptr<GenericIterativeSolve<M>> solver_ptr;

    Experiment_Data(
        Experiment_Clock arg_clock,
        std::shared_ptr<GenericIterativeSolve<M>> arg_solver_ptr
    ):
        clock(arg_clock), solver_ptr(arg_solver_ptr) 
    {}

    Experiment_Data(const Experiment_Data &other) = default;
    Experiment_Data & operator=(const Experiment_Data &other) = default;

    std::string get_info_string() const {
        return std::format(
            "{} | {}",
            clock.get_info_string(),
            solver_ptr->get_info_string()
        );
    }

};

std::string vector_to_jsonarray_str(std::vector<double> vec, int padding_level);

template <template <typename> typename M>
void record_experimental_data_json(
    const Experiment_Data<M> &data,
    const std::string ID,
    const fs::path save_dir
) {

    fs::path save_path(save_dir / fs::path(ID + ".json"));
    std::cout << std::format("Saving Experimental Data to: {}", save_path.string()) << std::endl;
    
    std::ofstream file_out;
    file_out.open(save_path, std::ofstream::out);

    if (file_out.is_open()) {

        file_out << std::scientific;
        file_out.precision(16);

        file_out << "{\n";

        file_out << std::format("\t\"id\" : \"{}\",\n", ID);
        file_out << std::format("\t\"solver_class\" : \"{}\",\n", typeid(*(data.solver_ptr)).name());
        file_out << std::format("\t\"initiated\" : \"{}\",\n", data.solver_ptr->check_initiated());
        file_out << std::format("\t\"converged\" : \"{}\",\n", data.solver_ptr->check_converged());
        file_out << std::format("\t\"terminated\" : \"{}\",\n", data.solver_ptr->check_terminated());
        file_out << std::format("\t\"res_norm_hist\" : {}\n",
            vector_to_jsonarray_str(data.solver_ptr->get_res_norm_hist(), 0)
        );

        file_out << "}";

        file_out.close();

    } else {
        throw std::runtime_error(
            "record_experimental_data_json: Failed to open for write: " + save_path.string()
        );
    }

}

#endif