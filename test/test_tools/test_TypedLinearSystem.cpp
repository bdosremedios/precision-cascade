#include "../test.h"

#include "tools/LinearSystem.h"

class TypedLinearSystem_Test: public TestBase
{
public:

    template <template <typename> typename M, typename T>
    void TestTypedConstructor() {

        constexpr int m(63);
        constexpr int n(27);
        M<double> A(M<double>::Random(*handle_ptr, m, n));
        MatrixVector<double> b(MatrixVector<double>::Random(*handle_ptr, m));
        TypedLinearSystem<M, T> lin_sys(A, b);

        ASSERT_MATRIX_EQ(lin_sys.get_A(), A);
        ASSERT_VECTOR_EQ(lin_sys.get_b(), b);

    }

};

TEST_F(TypedLinearSystem_Test, TestHalfConstructor) {
    TestTypedConstructor<MatrixDense, __half>();
    // TestTypedConstructor<MatrixSparse, __half>();
}

TEST_F(TypedLinearSystem_Test, TestSingleConstructor) {
    TestTypedConstructor<MatrixDense, float>();
    // TestTypedConstructor<MatrixSparse, float>();
}

TEST_F(TypedLinearSystem_Test, TestDoubleConstructor) {
    TestTypedConstructor<MatrixDense, double>();
    // TestTypedConstructor<MatrixSparse, double>();
}