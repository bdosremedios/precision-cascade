#include "../../test.h"

#include "tools/arg_pkgs/LinearSystem.h"

class TypedLinearSystem_Test: public TestBase
{
public:

    template <template <typename> typename M, typename T>
    void TestTypedConstructor() {

        constexpr int m(63);
        constexpr int n(27);
        M<double> A(CommonMatRandomInterface<M, double>::rand_matrix(TestBase::bundle, m, n));
        Vector<double> b(Vector<double>::Random(TestBase::bundle, m));

        GenericLinearSystem<M> gen_lin_sys(A, b);
        TypedLinearSystem<M, T> typed_lin_sys(&gen_lin_sys);

        EXPECT_EQ(typed_lin_sys.get_m(), m);
        EXPECT_EQ(typed_lin_sys.get_n(), n);
        EXPECT_EQ(typed_lin_sys.get_nnz(), A.non_zeros());
        EXPECT_EQ(typed_lin_sys.get_cu_handles(), TestBase::bundle);

        ASSERT_MATRIX_EQ(typed_lin_sys.get_A(), A);
        ASSERT_VECTOR_EQ(typed_lin_sys.get_b(), b);

        ASSERT_MATRIX_EQ(typed_lin_sys.get_A_typed(), A.template cast<T>());
        ASSERT_VECTOR_EQ(typed_lin_sys.get_b_typed(), b.template cast<T>());

    }

    template <template <typename> typename M, typename T>
    void TestTypedSetb() {

        constexpr int m(63);
        constexpr int n(27);
        M<double> A(CommonMatRandomInterface<M, double>::rand_matrix(TestBase::bundle, m, n));
        Vector<double> b(Vector<double>::Random(TestBase::bundle, m));
        Vector<double> new_b(Vector<double>::Random(TestBase::bundle, m));

        GenericLinearSystem<M> gen_lin_sys(A, b);
        TypedLinearSystem<M, T> typed_lin_sys(&gen_lin_sys);

        typed_lin_sys.set_b(new_b);

        ASSERT_VECTOR_EQ(typed_lin_sys.get_b(), new_b);
        ASSERT_VECTOR_EQ(typed_lin_sys.get_b_typed(), new_b.template cast<T>());

    }

    template <template <typename> typename M, typename T>
    void TestBadMismatchTypedSetb() {
        
        auto mismatch_setb_under = [=]() {
            constexpr int m(63);
            constexpr int n(27);
            M<double> A(CommonMatRandomInterface<M, double>::rand_matrix(TestBase::bundle, m, n));
            Vector<double> b(Vector<double>::Random(TestBase::bundle, n));

            GenericLinearSystem<M> gen_lin_sys(A, b);
            TypedLinearSystem<M, T> typed_lin_sys(&gen_lin_sys);

            Vector<double> bad_b(Vector<double>::Random(TestBase::bundle, n-1));
            typed_lin_sys.set_b(bad_b);
        };
        
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, mismatch_setb_under);
        
        auto mismatch_setb_over = [=]() {
            constexpr int m(63);
            constexpr int n(27);
            M<double> A(CommonMatRandomInterface<M, double>::rand_matrix(TestBase::bundle, m, n));
            Vector<double> b(Vector<double>::Random(TestBase::bundle, n));

            GenericLinearSystem<M> gen_lin_sys(A, b);
            TypedLinearSystem<M, T> typed_lin_sys(&gen_lin_sys);

            Vector<double> bad_b(Vector<double>::Random(TestBase::bundle, n+1));
            typed_lin_sys.set_b(bad_b);
        };
        
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, mismatch_setb_over);

    }

};

TEST_F(TypedLinearSystem_Test, TestHalfConstructor) {
    TestTypedConstructor<MatrixDense, __half>();
    TestTypedConstructor<NoFillMatrixSparse, __half>();
}

TEST_F(TypedLinearSystem_Test, TestSingleConstructor) {
    TestTypedConstructor<MatrixDense, float>();
    TestTypedConstructor<NoFillMatrixSparse, float>();
}

TEST_F(TypedLinearSystem_Test, TestDoubleConstructor) {
    TestTypedConstructor<MatrixDense, double>();
    TestTypedConstructor<NoFillMatrixSparse, double>();
}

TEST_F(TypedLinearSystem_Test, TestHalfTypedSetb) {
    TestTypedSetb<MatrixDense, __half>();
    TestTypedSetb<NoFillMatrixSparse, __half>();
}

TEST_F(TypedLinearSystem_Test, TestSingleTypedSetb) {
    TestTypedSetb<MatrixDense, float>();
    TestTypedSetb<NoFillMatrixSparse, float>();
}

TEST_F(TypedLinearSystem_Test, TestDoubleTypedSetb) {
    TestTypedSetb<MatrixDense, double>();
    TestTypedSetb<NoFillMatrixSparse, double>();
}

TEST_F(TypedLinearSystem_Test, TestHalfBadMismatchTypedSetb) {
    TestBadMismatchTypedSetb<MatrixDense, __half>();
    TestBadMismatchTypedSetb<NoFillMatrixSparse, __half>();
}

TEST_F(TypedLinearSystem_Test, TestSingleBadMismatchTypedSetb) {
    TestBadMismatchTypedSetb<MatrixDense, float>();
    TestBadMismatchTypedSetb<NoFillMatrixSparse, float>();
}

TEST_F(TypedLinearSystem_Test, TestDoubleBadMismatchTypedSetb) {
    TestBadMismatchTypedSetb<MatrixDense, double>();
    TestBadMismatchTypedSetb<NoFillMatrixSparse, double>();
}