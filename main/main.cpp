#include "tools/LinearSystem.h"

#include <iostream>

using std::cout, std::endl;

int main() {

    MatrixSparse<double> A = MatrixSparse<double>::Random(4, 4);
    
    for (int col=A.outerSize()-1; col>=0; --col) {
        for (MatrixSparse<double>::ReverseInnerIterator it(A, col); it; --it) {
            cout << it.row() << " " << it.col() << endl;
        }
    }

    return 0;

}