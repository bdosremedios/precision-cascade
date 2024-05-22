#include "../../test.h"

#include "tools/arg_pkgs/LinearSystem.h"

class GenericLinearSystem_Test: public TestBase
{
public:

    template <template <typename> typename M>
    void TestConstructor() {

        constexpr int m(63);
        constexpr int n(27);
        M<double> A(CommonMatRandomInterface<M, double>::rand_matrix(TestBase::bundle, m, n));
        Vector<double> b(Vector<double>::Random(TestBase::bundle, m));

        GenericLinearSystem<M> lin_sys(A, b);

        EXPECT_EQ(lin_sys.get_m(), m);
        EXPECT_EQ(lin_sys.get_n(), n);
        EXPECT_EQ(lin_sys.get_nnz(), A.non_zeros());
        EXPECT_EQ(lin_sys.get_cu_handles(), TestBase::bundle);

        ASSERT_MATRIX_EQ(lin_sys.get_A(), A);
        ASSERT_VECTOR_EQ(lin_sys.get_b(), b);

    }

    template <template <typename> typename M>
    void TestBadEmptyMatrix() {
        
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() {
                M<double> A(CommonMatRandomInterface<M, double>::rand_matrix(TestBase::bundle, 0, 0));
                Vector<double> b(Vector<double>::Random(TestBase::bundle, 0));
                GenericLinearSystem<M> lin_sys(A, b);
            }
        );

    }

    template <template <typename> typename M>
    void TestBadMismatchb() {
        
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() {
                constexpr int m(63);
                constexpr int n(27);
                M<double> A(CommonMatRandomInterface<M, double>::rand_matrix(TestBase::bundle, m, n));
                Vector<double> bad_b(Vector<double>::Random(TestBase::bundle, m-1));
                GenericLinearSystem<M> lin_sys(A, bad_b);
            }
        );
        
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() {
                constexpr int m(63);
                constexpr int n(27);
                M<double> A(CommonMatRandomInterface<M, double>::rand_matrix(TestBase::bundle, m, n));
                Vector<double> bad_b(Vector<double>::Random(TestBase::bundle, m+1));
                GenericLinearSystem<M> lin_sys(A, bad_b);
            }
        );

    }

};

TEST_F(GenericLinearSystem_Test, TestConstructor) {
    TestConstructor<MatrixDense>();
    TestConstructor<NoFillMatrixSparse>();
}

TEST_F(GenericLinearSystem_Test, TestBadEmptyMatrix) {
    TestBadEmptyMatrix<MatrixDense>();
    TestBadEmptyMatrix<NoFillMatrixSparse>();
}

TEST_F(GenericLinearSystem_Test, TestBadMismatchb) {
    TestBadMismatchb<MatrixDense>();
    TestBadMismatchb<NoFillMatrixSparse>();
}