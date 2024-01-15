#include "../test.h"

#include "tools/LinearSystem.h"

class GenericLinearSystem_Test: public TestBase
{
public:

    template <template <typename> typename M>
    void TestConstructor() {

        constexpr int m(63);
        constexpr int n(27);
        M<double> A(M<double>::Random(*handle_ptr, m, n));
        MatrixVector<double> b(MatrixVector<double>::Random(*handle_ptr, m));
        GenericLinearSystem<M> lin_sys(A, b);

        EXPECT_EQ(lin_sys.get_m(), m);
        EXPECT_EQ(lin_sys.get_n(), n);

        ASSERT_MATRIX_EQ(lin_sys.get_A(), A);
        ASSERT_VECTOR_EQ(lin_sys.get_b(), b);

    }

    template <template <typename> typename M>
    void TestEmptyMatrix() {

        try {
            M<double> A(M<double>::Random(*handle_ptr, 0, 0));
            MatrixVector<double> b(MatrixVector<double>::Random(*handle_ptr, 0));
            GenericLinearSystem<M> lin_sys(A, b);
            FAIL();
        } catch (std::runtime_error e) {
            std::cout << e.what() << std::endl;
        }

    }

    template <template <typename> typename M>
    void TestMismatchb() {

        try {
            constexpr int m(63);
            constexpr int n(27);
            M<double> A(M<double>::Random(*handle_ptr, m, n));
            MatrixVector<double> b(MatrixVector<double>::Random(*handle_ptr, n-1));
            GenericLinearSystem<M> lin_sys(A, b);
            FAIL();
        } catch (std::runtime_error e) {
            std::cout << e.what() << std::endl;
        }

    }

};

TEST_F(GenericLinearSystem_Test, TestConstructor) {
    TestConstructor<MatrixDense>();
    // TestConstructor<MatrixSparse>();
}

TEST_F(GenericLinearSystem_Test, TestEmptyMatrix) {
    TestEmptyMatrix<MatrixDense>();
    // TestEmptyMatrix<MatrixSparse>();
}

TEST_F(GenericLinearSystem_Test, TestMismatchb) {
    TestMismatchb<MatrixDense>();
    // TestMismatchb<MatrixSparse>();
}