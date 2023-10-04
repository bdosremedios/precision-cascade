#include "tools/LinearSystem.h"

#include <iostream>

using std::cout, std::endl;

template <template <typename> typename M>
class TestClass
{
public:
    const GenericLinearSystem<M> &genlinsys;
    
    TestClass(const GenericLinearSystem<M> &arg_genlinsys): 
        genlinsys(arg_genlinsys)
    {}

};

int main() {

    // MatrixDense<double> A = MatrixDense<double>::Random(4, 4);
    // MatrixVector<double> b = MatrixVector<double>::Random(4);

    // TypedLinearSystem<MatrixDense, double> typlinsys(A, b);

    // cout << typlinsys.A << endl << endl;

    // TestClass<MatrixDense> test(typlinsys);

    // cout << test.genlinsys.A << endl << endl;

    // typlinsys.A = MatrixDense<double>::Random(2, 2);

    // cout << typlinsys.A << endl << endl;

    // cout << test.genlinsys.A << endl << endl;

    return 0;

}