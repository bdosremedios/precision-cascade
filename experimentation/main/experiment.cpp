#include <memory>

#include <filesystem>
#include <iostream>
#include <sstream>

#include <cmath>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "experiment_log.h"
#include "experiment_tools.h"
#include "experiment_run.h"

namespace fs = std::filesystem;

int main(int argc, char *argv[]) {

    fs::path data_dir_path("C:\\Users\\dosre\\dev\\numerical_experimentation\\data");
    fs::path input_dir_path("C:\\Users\\dosre\\dev\\numerical_experimentation\\input");
    fs::path output_dir_path("C:\\Users\\dosre\\dev\\numerical_experimentation\\output");

    Experiment_Log experiment_logger(
        "experiment", output_dir_path / fs::path("experiment.log"), true
    );

    experiment_logger.info(std::format("Start numerical experiment: {}", argv[0]));
    experiment_logger.info("Data directory for experiment matrices: " + data_dir_path.string());
    experiment_logger.info("Input directory for experiment specifications: " + input_dir_path.string());
    experiment_logger.info("Output directory for experiment data: " + output_dir_path.string());

    fs::directory_iterator dir_iter = fs::directory_iterator(input_dir_path);

    // Find candidate experimental spec files in input directory (all jsons)
    std::vector<fs::path> candidate_exp_specs;
    experiment_logger.info(std::format("Searching {} for experimental spec files", input_dir_path.string()));
    for (auto curr = begin(dir_iter); curr != end(dir_iter); ++curr) {
        if (curr->path().extension() == ".json") {
            candidate_exp_specs.push_back(curr->path());
        }
    }
    experiment_logger.info(std::format("Found {} experimental spec files", candidate_exp_specs.size()));

    // Validate found experimental specs
    std::vector<Experiment_Specification> valid_exp_specs;
    experiment_logger.info("Validating experimental spec files");
    for (fs::path cand_exp_spec_path : candidate_exp_specs) {
        try {
            experiment_logger.info("Validating: " + cand_exp_spec_path.filename().string());
            Experiment_Specification loaded_exp_spec = parse_experiment_spec(cand_exp_spec_path);
            valid_exp_specs.push_back(loaded_exp_spec);
        } catch (std::runtime_error e) {
            experiment_logger.warn("Failed validation for: " + cand_exp_spec_path.filename().string());
            experiment_logger.warn("Skipping: " + cand_exp_spec_path.filename().string());
        }
    }
    experiment_logger.info(
        std::format(
            "Completed validation: {} passed | {} fail",
            valid_exp_specs.size(),
            candidate_exp_specs.size()-valid_exp_specs.size()
        )
    );

    // Set-up cublas context
    cuHandleBundle cu_handles;
    cu_handles.create();

    // Run valid experimental specs
    for (Experiment_Specification exp_spec : valid_exp_specs) {
        run_experimental_spec(cu_handles, exp_spec, data_dir_path, output_dir_path, experiment_logger);
    }

    cu_handles.destroy();

    experiment_logger.info("Finish numerical experiment");
    
    return 0;

}