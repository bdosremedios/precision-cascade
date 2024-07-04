#include "exp_run/exp_run_record.h"

void create_or_clear_directory(fs::path dir, Experiment_Log logger) {

    if (!fs::exists(dir)) {
        logger.info("Create dir: "+dir.string());
        fs::create_directory(dir);
    } else {
        logger.info("Clear dir: "+dir.string());
        for (auto member : fs::directory_iterator(dir)) {
            fs::remove_all(member);
        }
    }

}

void run_record_experimental_spec(
    const cascade::cuHandleBundle &cu_handles,
    Experiment_Spec exp_spec,
    fs::path matrix_data_dir,
    fs::path output_data_dir,
    Experiment_Log logger
) {

    logger.info("Start Experiment_Spec: " + exp_spec.id);

    fs::path exp_spec_dir = output_data_dir / fs::path(exp_spec.id);
    create_or_clear_directory(exp_spec_dir, logger);

    for (Solve_Group solve_group : exp_spec.solve_groups) {

        if (solve_group.matrix_type == "dense") {

            run_record_solve_group<cascade::MatrixDense>(
                cu_handles, solve_group, matrix_data_dir, exp_spec_dir, logger
            );

        } else if (solve_group.matrix_type == "sparse") {

            run_record_solve_group<cascade::NoFillMatrixSparse>(
                cu_handles, solve_group, matrix_data_dir, exp_spec_dir, logger
            );

        } else {

            throw std::runtime_error(
                "run_record_experimental_spec: error invalid Solve_Group "
                "matrix type"
            );

        }

    }

}