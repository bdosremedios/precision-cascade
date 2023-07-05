#ifndef MATRIXREADER_H
#define MATRIXREADER_H

#include "Eigen/Dense"
#include <string>

using Eigen::Matrix, Eigen::Dynamic;
using std::string;

namespace mxread
{

class MatrixReader {

    public:
        MatrixReader() = default;
        Matrix<double, Dynamic, Dynamic> read_file_d(string const &) const;

};

} // namespace mxread

#endif