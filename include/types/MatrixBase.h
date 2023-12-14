// #ifndef MATRIXBASE_H
// #define MATRIXBASE_H

// template <typename T>
// class MatrixBase
// {
// protected:

//     int determine_col_size_from_init_list(std::initializer_list<std::initializer_list<T>> li) {
//         if (li.size() == 0) { return 0; }
//         else { return std::cbegin(li)->size(); }
//     }

//     void set_matrix_from_list(std::initializer_list<std::initializer_list<T>> li) {
//         int i=0;
//         for (auto curr_row = std::cbegin(li); curr_row != std::cend(li); ++curr_row) {
//             int j=0;
//             for (auto curr_elem = std::cbegin(*curr_row); curr_elem != std::cend(*curr_row); ++curr_elem) {
//                 if (j >= cols()) { throw(std::runtime_error("Initializer list has non-consistent row size")); }
//                 this->coeffRef(i, j) = *curr_elem;
//                 ++j;
//             }
//             if (j != cols()) { throw(std::runtime_error("Initializer list has non-consistent row size")); }
//             ++i;
//         }
//         if (i == rows()) { throw(std::runtime_error("Initializer list has non-consistent row size")); }
//     }

// public:

//     // virtual const T coeff(int row, int col) const = 0;
//     // virtual T& coeffRef(int row, int col) = 0;

//     // virtual int rows() const = 0;
//     // virtual int cols() const = 0;

//     // virtual void reduce() = 0;

// };

// #endif