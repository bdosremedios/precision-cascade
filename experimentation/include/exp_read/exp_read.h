#ifndef EXP_READ_H
#define EXP_READ_H

#include "exp_spec/exp_spec.h"
#include "exp_tools/Experiment_Log.h"

#include "types/types.h"
#include "tools/cuHandleBundle.h"
#include "tools/read_matrix.h"
#include "tools/arg_pkgs/LinearSystem.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <string>
#include <vector>
#include <unordered_set>

namespace fs = std::filesystem;

using json = nlohmann::json;

int extract_integer(json::iterator member);
std::vector<std::string> extract_solvers_to_use(json::iterator member);
std::string extract_matrix_type(json::iterator member);
Preconditioner_Spec extract_preconditioner_spec(
    json::iterator member
);
double extract_double(json::iterator member);
std::vector<std::string> extract_string_vector(json::iterator member);

Solve_Group extract_solve_group(std::string id, json cand_obj);

Experiment_Spec parse_experiment_spec(fs::path exp_spec_path);

template <template <typename> typename TMatrix>
cascade::GenericLinearSystem<TMatrix> load_lin_sys(
    const cascade::cuHandleBundle &cu_handles,
    fs::path input_dir,
    std::string matrix_name,
    Experiment_Log logger
) {

    fs::path matrix_path = input_dir / fs::path(matrix_name);

    logger.info("Loading: "+matrix_path.string());

    TMatrix<double> A(cu_handles);
    if (matrix_path.extension() == ".mtx") {
        A = cascade::read_matrixMTX<TMatrix, double>(cu_handles, matrix_path);
    } else if (matrix_path.extension() == ".csv") {
        A = cascade::read_matrixCSV<TMatrix, double>(cu_handles, matrix_path);
    } else {
        throw std::runtime_error("load_lin_sys: invalid extension");
    }

    cascade::Scalar<double> A_max_mag = A.get_max_mag_elem();
    A /= A_max_mag;
    logger.info("Matrix info: " + A.get_info_string());

    // Search for a rhs and if none is found generate one randomly
    fs::path potential_b_path(
        input_dir /
        fs::path(
            matrix_path.stem().string() + "_b" +
            matrix_path.extension().string()
        )
    );
    cascade::Vector<double> b(cu_handles);
    if (fs::exists(potential_b_path)) {
        if (potential_b_path.extension() == ".mtx") {
            b = cascade::read_vectorMTX<double>(
                cu_handles, potential_b_path, "random"
            );
        } else if (potential_b_path.extension() == ".csv") {
            b = cascade::read_vectorCSV<double>(
                cu_handles, potential_b_path
            );
        } else {
            throw std::runtime_error(
                "load_lin_sys: invalid extension found on potential_b_path file"
            );
        }
        b /= A_max_mag;
    } else {
        b = A*cascade::Vector<double>::Random(cu_handles, A.cols());
    }

    return cascade::GenericLinearSystem<TMatrix>(A, b);

}

#endif