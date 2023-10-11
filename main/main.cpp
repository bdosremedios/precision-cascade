#include "tools/LinearSystem.h"

#include <iostream>

using std::cout, std::endl;

int main() {

    MatrixSparse<double> A = MatrixSparse<double>::Zero(4, 4);
    A.prune(0.);

    using LowerTriag = const Eigen::TriangularView<const SparseMatrix<double>, Eigen::Lower>;

    LowerTriag buddy(A.triangularView<Eigen::Lower>());

    // for (LowerTriag::Iterator it; it; ++it) {
    //     cout << it << endl;
    // }
    // auto it = buddy.col(0)::iterator;




    return 0;

}