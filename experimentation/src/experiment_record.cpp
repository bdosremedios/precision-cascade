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