#include "../test.h"

class GenericLinearSystemTest: public TestBase
{
public:

    template <template <typename> typename M>
    void TestConstructor() {

        constexpr int m(63);
        constexpr int n(27);
        M<double> A = M<double>::Random(m, n);
        MatrixVector<double> b = MatrixVector<double>::Random(m);
        GenericLinearSystem<M> lin_sys(A, b);

        EXPECT_EQ(lin_sys.get_m(), m);
        EXPECT_EQ(lin_sys.get_n(), n);

        EXPECT_EQ(lin_sys.get_A(), A);
        EXPECT_EQ(lin_sys.get_b(), b);

    }

    template <template <typename> typename M>
    void TestEmptyMatrix() {

        try {
            M<double> A = M<double>::Random(0, 0);
            MatrixVector<double> b = MatrixVector<double>::Random(0);
            GenericLinearSystem<M> lin_sys(A, b);
            FAIL();
        } catch (runtime_error e) {
            cout << e.what() << endl;
        }

    }

    template <template <typename> typename M>
    void TestMismatchb() {

        try {
            constexpr int m(63);
            constexpr int n(27);
            M<double> A = M<double>::Random(m, n);
            MatrixVector<double> b = MatrixVector<double>::Random(n-1);
            GenericLinearSystem<M> lin_sys(A, b);
            FAIL();
        } catch (runtime_error e) {
            cout << e.what() << endl;
        }

    }

};

TEST_F(GenericLinearSystemTest, TestConstructor_Dense) { TestConstructor<MatrixDense>(); }
TEST_F(GenericLinearSystemTest, TestConstructor_Sparse) { TestConstructor<MatrixSparse>(); }

TEST_F(GenericLinearSystemTest, TestEmptyMatrix_Dense) { TestEmptyMatrix<MatrixDense>(); }
TEST_F(GenericLinearSystemTest, TestEmptyMatrix_Sparse) { TestEmptyMatrix<MatrixSparse>(); }

TEST_F(GenericLinearSystemTest, TestMismatchb_Dense) { TestMismatchb<MatrixDense>(); }
TEST_F(GenericLinearSystemTest, TestMismatchb_Sparse) { TestMismatchb<MatrixSparse>(); }