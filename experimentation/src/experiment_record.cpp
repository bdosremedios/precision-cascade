#include "experiment_record.h"

std::string vector_to_jsonarray_str(
    std::vector<double> vec, int padding_level
) {

    std::string str_to_write = "";
    for (int i=0; i<padding_level; ++i) { str_to_write += "\t"; }
    str_to_write += "[";
    for (int i=0; i<vec.size()-1; ++i) {
        str_to_write += std::format("{:.16e},", vec[i]);
    }
    str_to_write += std::format("{:.16e}", vec[vec.size()-1]);
    str_to_write += "]";

    return str_to_write;

}

std::string matrix_to_jsonarray_str(
    MatrixDense<double> mat, int padding_level
) {

    std::string str_to_write = "";
    str_to_write += "[\n";
    for (int i=0; i<mat.rows()-1; ++i) {
        for (int i=0; i<padding_level+1; ++i) { str_to_write += "\t"; }
        str_to_write += "[";
        for (int j=0; j<mat.cols()-1; ++j) {
            str_to_write += std::format(
                "{:.16e},",
                mat.get_elem(i, j).get_scalar()
            );
        }
        str_to_write += std::format(
            "{:.16e}],\n",
            mat.get_elem(i, mat.cols()-1).get_scalar()
        );
    }
    for (int i=0; i<padding_level+1; ++i) { str_to_write += "\t"; }
    str_to_write += "[";
    for (int j=0; j<mat.cols()-1; ++j) {
        str_to_write += std::format(
            "{:.16e},",
            mat.get_elem(mat.rows()-1, j).get_scalar()
        );
    }
    str_to_write += std::format(
        "{:.16e}]\n",
        mat.get_elem(mat.rows()-1, mat.cols()-1).get_scalar()
    );
    for (int i=0; i<padding_level; ++i) { str_to_write += "\t"; }
    str_to_write += "]";

    return str_to_write;

}