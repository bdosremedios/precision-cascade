#include "exp_data/Timed_Experiment_Data.h"

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