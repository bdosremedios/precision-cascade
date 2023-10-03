#include "../test.h"

class TypedLinearSystemTest: public TestBase
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

TEST_F(TypedLinearSystemTest, TestHalfConstructor_Dense) { TestTypedConstructor<MatrixDense, half>(); }
TEST_F(TypedLinearSystemTest, TestHalfConstructor_Sparse) { TestTypedConstructor<MatrixSparse, half>(); }

TEST_F(TypedLinearSystemTest, TestSingleConstructor_Dense) { TestTypedConstructor<MatrixDense, float>(); }
TEST_F(TypedLinearSystemTest, TestSingleConstructor_Sparse) { TestTypedConstructor<MatrixSparse, float>(); }

TEST_F(TypedLinearSystemTest, TestDoubleConstructor_Dense) { TestTypedConstructor<MatrixDense, double>(); }
TEST_F(TypedLinearSystemTest, TestDoubleConstructor_Sparse) { TestTypedConstructor<MatrixSparse, double>(); }