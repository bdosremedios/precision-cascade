#ifndef EXPERIMENT_RECORD_H
#define EXPERIMENT_RECORD_H

#include <chrono>
#include <string>
#include <format>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <filesystem>

#include "solvers/IterativeSolve.h"
#include "solvers/nested/GMRES_IR/MP_GMRES_IR.h"

#include "experiment_log.h"
#include "experiment_tools.h"

namespace fs = std::filesystem;

std::string vector_to_jsonarray_str(std::vector<double> vec, int padding_level);

std::string matrix_to_jsonarray_str(MatrixDense<double> mat, int padding_level);

template <template <typename> typename M>
void record_basic_solver_data(
    std::ofstream &file_out,
    const std::string ID,
    const std::shared_ptr<GenericIterativeSolve<M>> &solver_ptr,
    const Experiment_Clock &clock
) {
    file_out << std::format("\t\"id\" : \"{}\",\n", ID);
    file_out << std::format("\t\"solver_class\" : \"{}\",\n", typeid(*solver_ptr).name());
    file_out << std::format("\t\"initiated\" : \"{}\",\n", solver_ptr->check_initiated());
    file_out << std::format("\t\"converged\" : \"{}\",\n", solver_ptr->check_converged());
    file_out << std::format("\t\"terminated\" : \"{}\",\n", solver_ptr->check_terminated());
    file_out << std::format("\t\"iteration\" : {},\n", solver_ptr->get_iteration());
    file_out << std::format("\t\"elapsed_time_ms\" : {},\n", clock.get_elapsed_time_ms());
}

template <template <typename> typename M, typename T>
void record_precond_data(
    std::ofstream &file_out,
    const PrecondArgPkg<M, T> arg_precond_arg_pkg,
    const std::string precond_specs_str
) {
    file_out << std::format(
        "\t\"precond_class\" : \"{}\",\n", typeid(arg_precond_arg_pkg).name()
    );
    file_out << std::format(
        "\t\"precond_left\" : \"{}\",\n", typeid(*arg_precond_arg_pkg.left_precond).name()
    );
    file_out << std::format(
        "\t\"precond_right\" : \"{}\",\n", typeid(*arg_precond_arg_pkg.right_precond).name()
    );
    file_out << std::format(
        "\t\"precond_specs\" : \"{}\",\n", precond_specs_str
    );
}

template <template <typename> typename M>
void record_residual_solver_data(
    std::ofstream &file_out,
    const std::shared_ptr<GenericIterativeSolve<M>> &solver_ptr,
    const int padding
) {
    file_out << std::format("\t\"res_norm_history\" : {}\n",
        vector_to_jsonarray_str(solver_ptr->get_res_norm_history(), padding)
    );
}

template <template <typename> typename M, typename T>
void record_FPGMRES_data_json(
    const Experiment_Data<GenericIterativeSolve, M> &data,
    const PrecondArgPkg<M, T> arg_precond_arg_pkg,
    const std::string precond_specs_str,
    const std::string file_name,
    const fs::path save_dir,
    Experiment_Log logger
) {

    fs::path save_path(save_dir / fs::path(file_name + ".json"));
    logger.info(std::format("Save data to: {}", save_path.string()));
    
    std::ofstream file_out;
    file_out.open(save_path, std::ofstream::out);

    if (file_out.is_open()) {

        file_out << "{\n";

        record_basic_solver_data<M>(file_out, file_name, data.solver_ptr, data.clock);
        record_precond_data<M, T>(file_out, arg_precond_arg_pkg, precond_specs_str);
        record_residual_solver_data<M>(file_out, data.solver_ptr, 0);

        file_out << "}";

        file_out.close();

    } else {
        throw std::runtime_error(
            "record_experimental_data_json: Failed to open for write: " + save_path.string()
        );
    }

}

template <template <typename> typename M>
void record_MPGMRES_data_json(
    const Experiment_Data<MP_GMRES_IR_Solve, M> &data,
    const PrecondArgPkg<M, double> arg_precond_arg_pkg,
    const std::string precond_specs_str,
    const std::string file_name,
    const fs::path save_dir,
    Experiment_Log logger
) {

    fs::path save_path(save_dir / fs::path(file_name + ".json"));
    logger.info(std::format("Save data to: {}", save_path.string()));
    
    std::ofstream file_out;
    file_out.open(save_path, std::ofstream::out);

    if (file_out.is_open()) {

        file_out << "{\n";

        record_basic_solver_data<M>(file_out, file_name, data.solver_ptr, data.clock);
        file_out << std::format(
            "\t\"hlf_sgl_cascade_change\" : \"{}\",\n", data.solver_ptr->get_hlf_sgl_cascade_change()
        );
        file_out << std::format(
            "\t\"sgl_dbl_cascade_change\" : \"{}\",\n", data.solver_ptr->get_sgl_dbl_cascade_change()
        );
        record_precond_data<M, double>(file_out, arg_precond_arg_pkg, precond_specs_str);
        record_residual_solver_data<M>(file_out, data.solver_ptr, 0);

        file_out << "}";

        file_out.close();

    } else {
        throw std::runtime_error(
            "record_MPGMRES_experimental_data_json: Failed to open for write: " + save_path.string()
        );
    }

}

#endif