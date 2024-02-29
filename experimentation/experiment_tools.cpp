#include "experiment_tools.h"

#include <string>

void Experiment_Clock::start_clock_experiment() {
    if (!clock_ticking) {
        start = clock.now();
        clock_ticking = true;
    } else {
        throw std::runtime_error("Experiment_Clock: start_clock_experiment clock already ticking");
    } 
}
    
void Experiment_Clock::stop_clock_experiment() {
    if (clock_ticking) {
        stop = clock.now();
        clock_ticking = false;
        time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop-start);
    } else {
        throw std::runtime_error("Experiment_Clock: stop_clock_experiment clock not ticking");
    }
}

int Experiment_Clock::get_elapsed_time_ms() const {
    return time_ms.count();
}

std::string Experiment_Clock::get_info_string() const {
    return std::format("Elapsed time (ms): {}", get_elapsed_time_ms());
}

std::string vector_to_jsonarray_str(std::vector<double> vec, int padding_level) {

    std::string str_to_write = "";
    for (int i=0; i<padding_level; ++i) { str_to_write += "\t"; }
    str_to_write += "[";
    for (int i=0; i<vec.size()-1; ++i) {
        str_to_write += std::format("{:15e},", vec[i]);
    }
    str_to_write += std::format("{:15e}", vec[vec.size()-1]);
    str_to_write += "]";

    return str_to_write;

}

std::string matrix_to_jsonarray_str(MatrixDense<double> mat, int padding_level) {

    std::string str_to_write = "";
    str_to_write += "[\n";
    for (int i=0; i<mat.rows()-1; ++i) {
        for (int i=0; i<padding_level+1; ++i) { str_to_write += "\t"; }
        str_to_write += "[";
        for (int j=0; j<mat.cols()-1; ++j) {
            str_to_write += std::format("{:15e},", mat.get_elem(i, j).get_scalar());
        }
        str_to_write += std::format("{:15e}],\n", mat.get_elem(i, mat.cols()-1).get_scalar());
    }
    for (int i=0; i<padding_level+1; ++i) { str_to_write += "\t"; }
    str_to_write += "[";
    for (int j=0; j<mat.cols()-1; ++j) {
        str_to_write += std::format("{:15e},", mat.get_elem(mat.rows()-1, j).get_scalar());
    }
    str_to_write += std::format("{:15e}]\n", mat.get_elem(mat.rows()-1, mat.cols()-1).get_scalar());
    for (int i=0; i<padding_level; ++i) { str_to_write += "\t"; }
    str_to_write += "]";

    return str_to_write;

}