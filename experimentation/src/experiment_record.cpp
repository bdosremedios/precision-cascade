#include "experiment_record.h"

std::string vector_to_jsonarray_str(
    std::vector<double> vec, int padding_level
) {

    std::stringstream strm_to_write;
    strm_to_write << std::setprecision(17);
    for (int i=0; i<padding_level; ++i) { strm_to_write << "\t"; }
    strm_to_write << "[";
    for (int i=0; i<vec.size()-1; ++i) {
        strm_to_write << vec[i] << ", ";
    }
    strm_to_write << vec[vec.size()-1] << "]";

    return strm_to_write.str();

}

std::string bool_to_string(bool b) {
    if (b) {
        return "true";
    } else {
        return "false";
    }
}

std::ofstream open_file_ofstream(
    std::string file_name,
    fs::path save_dir,
    Experiment_Log logger
) {
    
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

void start_json(std::ofstream &file_out) {
    file_out << "{\n";
}

void end_json(std::ofstream &file_out) {
    file_out << "}";
    file_out.close();
}