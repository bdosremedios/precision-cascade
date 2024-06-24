#include "experiment_log.h"
#include "experiment_tools.h"
#include "experiment_run.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <memory>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <cmath>
#include <string>

namespace fs = std::filesystem;

int main(int argc, char *argv[]) {

    fs::path data_dir_path;
    fs::path input_dir_path;
    fs::path output_dir_path;

    if (
        (argc == 2) &&
        ((std::string(argv[1]) == "--help") || (std::string(argv[1]) == "-h"))
    ) {

        std::cout << "---- Entry point of precision-cascade experimentation ---"
                     "-"
                  << std::endl;
        std::cout << "REQUIRES: \"matrix_data\", \"input_specs\", and "
                     "\"output_data\" directories in same directory as "
                     "executable"
                  << std::endl;
        std::cout << "- \"matrix_data\" is the directory of the data matrix "
                     "csv and mtx files"
                  << std::endl;
        std::cout << "- \"input_specs\" is the directory experimental "
                     "spec json files to run"
                  << std::endl;
        std::cout << "- \"output_data\" is the directory to store output "
                     "experimental data in"
                  << std::endl;
        return EXIT_SUCCESS;

    } else {

        // Assumes data directories are in the same directory as executable
        #ifdef WIN32
            std::cout << fs::canonical("/proc/self/exe") << std::endl;
        #else
            fs::path exe_dir_path(
                fs::canonical("/proc/self/exe").parent_path()
            );
            data_dir_path = exe_dir_path / fs::path("matrix_data");
            input_dir_path = exe_dir_path / fs::path("input_specs");
            output_dir_path = exe_dir_path / fs::path("output_data");
        #endif

        // Check existence of directories
        if (!fs::is_directory(data_dir_path)) {
            std::cerr << "Invalid matrix data directory"
                      << std::endl;
            return EXIT_FAILURE;
        }
        if (!fs::is_directory(input_dir_path)) {
            std::cerr << "Invalid input experimental spec directory"
                      << std::endl;
            return EXIT_FAILURE;
        }
        if (!fs::is_directory(output_dir_path)) {
            std::cerr << "Invalid output experimental data directory"
                      << std::endl;
            return EXIT_FAILURE;
        }

    }

    Experiment_Log experiment_logger(
        "experiment", output_dir_path / fs::path("experiment.log"), true
    );

    experiment_logger.info(
        "Start numerical experiment: " + static_cast<std::string>(argv[0])
    );
    experiment_logger.info(
        "Data directory for experiment matrices: " +
        data_dir_path.string()
    );
    experiment_logger.info(
        "Input directory for experiment specifications: " +
        input_dir_path.string()
    );
    experiment_logger.info(
        "Output directory for experiment data: " +
        output_dir_path.string()
    );

    fs::directory_iterator dir_iter = fs::directory_iterator(input_dir_path);

    // Find candidate experimental spec files in input directory (all jsons)
    std::vector<fs::path> candidate_exp_specs;
    experiment_logger.info(
        "Searching {} for experimental spec files" + input_dir_path.string()
    );
    for (auto curr = begin(dir_iter); curr != end(dir_iter); ++curr) {
        if (curr->path().extension() == ".json") {
            candidate_exp_specs.push_back(curr->path());
        }
    }
    experiment_logger.info(
        "Found {} experimental spec files" +
        std::to_string(candidate_exp_specs.size())
    );

    // Extract and validate found experimental specs
    std::vector<Experiment_Specification> valid_exp_specs;
    experiment_logger.info("Validating experimental spec files");
    for (fs::path cand_exp_spec_path : candidate_exp_specs) {
        try {
            experiment_logger.info(
                "Validating: " + cand_exp_spec_path.filename().string()
            );
            Experiment_Specification loaded_exp_spec = parse_experiment_spec(
                cand_exp_spec_path
            );
            valid_exp_specs.push_back(loaded_exp_spec);
        } catch (std::runtime_error e) {
            experiment_logger.warn(
                "Failed validation for: " +
                cand_exp_spec_path.filename().string() +
                " with runtime_error " +
                e.what()
            );
            experiment_logger.warn(
                "Skipping: " + cand_exp_spec_path.filename().string()
            );
        }
    }
    experiment_logger.info(
        "Completed validation: " + std::to_string(valid_exp_specs.size()) + 
        " passed | " +
        std::to_string(candidate_exp_specs.size()-valid_exp_specs.size()) +
         " fail"
    );

    // Set-up cublas context
    cuHandleBundle cu_handles;
    cu_handles.create();

    // Run valid experimental specs
    for (Experiment_Specification exp_spec : valid_exp_specs) {
        try {
            run_experimental_spec(
                cu_handles,
                exp_spec,
                data_dir_path,
                output_dir_path,
                experiment_logger
            );
        } catch (std::runtime_error e) {
            experiment_logger.warn(
                "Failed running of for: " + exp_spec.id +
                " with runtime_error " + e.what()
            );
        }
    }

    cu_handles.destroy();

    experiment_logger.info("Finish numerical experiment");
    
    return EXIT_SUCCESS;

}