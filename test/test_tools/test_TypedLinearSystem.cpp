#include "../test.h"

class TypedLinearSystem_Test: public TestBase
{
public:

    template <template <typename> typename M, typename T>
    void TestTypedConstructor() {

        constexpr int m(63);
        constexpr int n(27);
        M<double> A = M<double>::Random(m, n);
        MatrixVector<double> b = MatrixVector<double>::Random(m);
        TypedLinearSystem<M, T> lin_sys(A, b);

        EXPECT_EQ(lin_sys.get_A(), A);
        EXPECT_EQ(lin_sys.get_b(), b);

    }

};

TEST_F(TypedLinearSystem_Test, TestHalfConstructor_Dense) { TestTypedConstructor<MatrixDense, half>(); }
TEST_F(TypedLinearSystem_Test, TestHalfConstructor_Sparse) { TestTypedConstructor<MatrixSparse, half>(); }

TEST_F(TypedLinearSystem_Test, TestSingleConstructor_Dense) { TestTypedConstructor<MatrixDense, float>(); }
TEST_F(TypedLinearSystem_Test, TestSingleConstructor_Sparse) { TestTypedConstructor<MatrixSparse, float>(); }

TEST_F(TypedLinearSystem_Test, TestDoubleConstructor_Dense) { TestTypedConstructor<MatrixDense, double>(); }
TEST_F(TypedLinearSystem_Test, TestDoubleConstructor_Sparse) { TestTypedConstructor<MatrixSparse, double>(); }