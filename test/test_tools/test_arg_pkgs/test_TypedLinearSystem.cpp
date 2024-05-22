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
    void TestTypedMutAddlRHSConstructor() {

        constexpr int m(63);
        constexpr int n(27);
        M<double> A(CommonMatRandomInterface<M, double>::rand_matrix(TestBase::bundle, m, n));
        Vector<double> b(Vector<double>::Random(TestBase::bundle, m));
        Vector<double> additional_rhs(Vector<double>::Random(TestBase::bundle, m));

        GenericLinearSystem<M> gen_lin_sys(A, b);
        TypedLinearSystem<M, T> temp(&gen_lin_sys);
        TypedLinearSystem_MutAddlRHS typed_lin_sys(&temp, additional_rhs);

        EXPECT_EQ(typed_lin_sys.get_m(), m);
        EXPECT_EQ(typed_lin_sys.get_n(), n);
        EXPECT_EQ(typed_lin_sys.get_nnz(), A.non_zeros());
        EXPECT_EQ(typed_lin_sys.get_cu_handles(), TestBase::bundle);

        ASSERT_MATRIX_EQ(typed_lin_sys.get_A(), A);
        ASSERT_VECTOR_EQ(typed_lin_sys.get_b(), additional_rhs);

        ASSERT_MATRIX_EQ(typed_lin_sys.get_A_typed(), A.template cast<T>());
        ASSERT_VECTOR_EQ(typed_lin_sys.get_b_typed(), additional_rhs.template cast<T>());

    }


    template <template <typename> typename M, typename T>
    void TestTypedMutAddlRHSSetRHS() {

        constexpr int m(63);
        constexpr int n(27);
        M<double> A(CommonMatRandomInterface<M, double>::rand_matrix(TestBase::bundle, m, n));
        Vector<double> b(Vector<double>::Random(TestBase::bundle, m));
        Vector<double> additional_rhs(Vector<double>::Random(TestBase::bundle, m));
        Vector<double> new_rhs(Vector<double>::Random(TestBase::bundle, m));

        GenericLinearSystem<M> gen_lin_sys(A, b);
        TypedLinearSystem<M, T> temp(&gen_lin_sys);
        TypedLinearSystem_MutAddlRHS typed_lin_sys(&temp, additional_rhs);
        
        // Check b has changed
        typed_lin_sys.set_rhs(new_rhs);
        ASSERT_VECTOR_EQ(typed_lin_sys.get_b(), new_rhs);
        ASSERT_VECTOR_EQ(typed_lin_sys.get_b_typed(), new_rhs.template cast<T>());

        // Check remaining values are unchanged
        EXPECT_EQ(typed_lin_sys.get_m(), m);
        EXPECT_EQ(typed_lin_sys.get_n(), n);
        EXPECT_EQ(typed_lin_sys.get_nnz(), A.non_zeros());
        EXPECT_EQ(typed_lin_sys.get_cu_handles(), TestBase::bundle);
        ASSERT_MATRIX_EQ(typed_lin_sys.get_A(), A);
        ASSERT_MATRIX_EQ(typed_lin_sys.get_A_typed(), A.template cast<T>());

    }

    template <template <typename> typename M, typename T>
    void TestTypedMutAddlRHSBadMismatchSetRHS() {
        
        auto mismatch_setrhs_under = [=]() {

            constexpr int m(63);
            constexpr int n(27);
            M<double> A(CommonMatRandomInterface<M, double>::rand_matrix(TestBase::bundle, m, n));
            Vector<double> b(Vector<double>::Random(TestBase::bundle, m));
            Vector<double> additional_rhs(Vector<double>::Random(TestBase::bundle, m));

            GenericLinearSystem<M> gen_lin_sys(A, b);
            TypedLinearSystem<M, T> temp(&gen_lin_sys);
            TypedLinearSystem_MutAddlRHS typed_lin_sys(&temp, additional_rhs);

            Vector<double> bad_rhs(Vector<double>::Random(TestBase::bundle, m-1));
            typed_lin_sys.set_rhs(bad_rhs);

        };
        
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, mismatch_setrhs_under);
        
        auto mismatch_setrhs_over = [=]() {

            constexpr int m(63);
            constexpr int n(27);
            M<double> A(CommonMatRandomInterface<M, double>::rand_matrix(TestBase::bundle, m, n));
            Vector<double> b(Vector<double>::Random(TestBase::bundle, m));
            Vector<double> additional_rhs(Vector<double>::Random(TestBase::bundle, m));

            GenericLinearSystem<M> gen_lin_sys(A, b);
            TypedLinearSystem<M, T> temp(&gen_lin_sys);
            TypedLinearSystem_MutAddlRHS typed_lin_sys(&temp, additional_rhs);

            Vector<double> bad_rhs(Vector<double>::Random(TestBase::bundle, m+1));
            typed_lin_sys.set_rhs(bad_rhs);

        };
        
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, mismatch_setrhs_over);

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

TEST_F(TypedLinearSystem_Test, TestHalfTypedMutAddlRHSConstructor) {
    TestTypedMutAddlRHSConstructor<MatrixDense, __half>();
    TestTypedMutAddlRHSConstructor<NoFillMatrixSparse, __half>();
}

TEST_F(TypedLinearSystem_Test, TestSingleTypedMutAddlRHSConstructor) {
    TestTypedMutAddlRHSConstructor<MatrixDense, float>();
    TestTypedMutAddlRHSConstructor<NoFillMatrixSparse, float>();
}

TEST_F(TypedLinearSystem_Test, TestDoubleTypedMutAddlRHSConstructor) {
    TestTypedMutAddlRHSConstructor<MatrixDense, double>();
    TestTypedMutAddlRHSConstructor<NoFillMatrixSparse, double>();
}

TEST_F(TypedLinearSystem_Test, TestHalfMutAddlRHSSetRHS) {
    TestTypedMutAddlRHSSetRHS<MatrixDense, __half>();
    TestTypedMutAddlRHSSetRHS<NoFillMatrixSparse, __half>();
}

TEST_F(TypedLinearSystem_Test, TestSingleMutAddlRHSSetRHS) {
    TestTypedMutAddlRHSSetRHS<MatrixDense, float>();
    TestTypedMutAddlRHSSetRHS<NoFillMatrixSparse, float>();
}

TEST_F(TypedLinearSystem_Test, TestDoubleMutAddlRHSSetRHS) {
    TestTypedMutAddlRHSSetRHS<MatrixDense, double>();
    TestTypedMutAddlRHSSetRHS<NoFillMatrixSparse, double>();
}

TEST_F(TypedLinearSystem_Test, TestHalfTypedMutAddlRHSBadMismatchSetRHS) {
    TestTypedMutAddlRHSBadMismatchSetRHS<MatrixDense, __half>();
    TestTypedMutAddlRHSBadMismatchSetRHS<NoFillMatrixSparse, __half>();
}

TEST_F(TypedLinearSystem_Test, TestSingleTypedMutAddlRHSBadMismatchSetRHS) {
    TestTypedMutAddlRHSBadMismatchSetRHS<MatrixDense, float>();
    TestTypedMutAddlRHSBadMismatchSetRHS<NoFillMatrixSparse, float>();
}

TEST_F(TypedLinearSystem_Test, TestDoubleTypedMutAddlRHSBadMismatchSetRHS) {
    TestTypedMutAddlRHSBadMismatchSetRHS<MatrixDense, double>();
    TestTypedMutAddlRHSBadMismatchSetRHS<NoFillMatrixSparse, double>();
}