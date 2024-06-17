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

// std::string matrix_to_jsonarray_str(
//     MatrixDense<double> mat, int padding_level
// ) {

//     std::stringstream strm_to_write;
//     strm_to_write << std::setprecision(17);
//     strm_to_write << "[\n";
//     for (int i=0; i<mat.rows()-1; ++i) {
//         for (int i=0; i<padding_level+1; ++i) { strm_to_write << "\t"; }
//         strm_to_write << "[";
//         for (int j=0; j<mat.cols()-1; ++j) {
//             strm_to_write << mat.get_elem(i, j).get_scalar() << ",";
//         }
//         strm_to_write << mat.get_elem(i, mat.cols()-1).get_scalar() << "],\n";
//     }
//     for (int i=0; i<padding_level+1; ++i) { strm_to_write << "\t"; }
//     strm_to_write << "[";
//     for (int j=0; j<mat.cols()-1; ++j) {
//         strm_to_write << mat.get_elem(mat.rows()-1, j).get_scalar() << ",";
//     }
//     strm_to_write << mat.get_elem(mat.rows()-1, mat.cols()-1).get_scalar()
//                   << "]\n";
//     for (int i=0; i<padding_level; ++i) { strm_to_write << "\t"; }
//     strm_to_write <<  "]";

//     return strm_to_write.str();

// }