#include "test.h"

#include "tools/arg_pkgs/LinearSystem.h"

class TypedLinearSystem_Test: public TestBase
{
public:

    template <template <typename> typename TMatrix, typename TPrecision>
    void TestTypedConstructor() {

        constexpr int m(63);
        constexpr int n(27);
        TMatrix<double> A(
            CommonMatRandomInterface<TMatrix, double>::rand_matrix(
                TestBase::bundle, m, n
            )
        );
        Vector<double> b(Vector<double>::Random(TestBase::bundle, m));

        GenericLinearSystem<TMatrix> gen_lin_sys(A, b);
        TypedLinearSystem<TMatrix, TPrecision> typed_lin_sys(&gen_lin_sys);

        EXPECT_EQ(typed_lin_sys.get_m(), m);
        EXPECT_EQ(typed_lin_sys.get_n(), n);
        EXPECT_EQ(typed_lin_sys.get_nnz(), A.non_zeros());
        EXPECT_EQ(typed_lin_sys.get_cu_handles(), TestBase::bundle);

        ASSERT_MATRIX_EQ(typed_lin_sys.get_A(), A);
        ASSERT_VECTOR_EQ(typed_lin_sys.get_b(), b);

        ASSERT_MATRIX_EQ(
            typed_lin_sys.get_A_typed(),
            A.template cast<TPrecision>()
        );
        ASSERT_VECTOR_EQ(
            typed_lin_sys.get_b_typed(),
            b.template cast<TPrecision>()
        );

    }

    template <template <typename> typename TMatrix, typename TPrecision>
    void TestTypedMutAddlRHSConstructor() {

        constexpr int m(63);
        constexpr int n(27);
        TMatrix<double> A(
            CommonMatRandomInterface<TMatrix, double>::rand_matrix(
                TestBase::bundle, m, n
            )
        );
        Vector<double> b(
            Vector<double>::Random(TestBase::bundle, m)
        );
        Vector<double> additional_rhs(
            Vector<double>::Random(TestBase::bundle, m)
        );

        GenericLinearSystem<TMatrix> gen_lin_sys(A, b);
        TypedLinearSystem<TMatrix, TPrecision> temp(&gen_lin_sys);
        TypedLinearSystem_MutAddlRHS typed_lin_sys(&temp, additional_rhs);

        EXPECT_EQ(typed_lin_sys.get_m(), m);
        EXPECT_EQ(typed_lin_sys.get_n(), n);
        EXPECT_EQ(typed_lin_sys.get_nnz(), A.non_zeros());
        EXPECT_EQ(typed_lin_sys.get_cu_handles(), TestBase::bundle);

        ASSERT_MATRIX_EQ(typed_lin_sys.get_A(), A);
        ASSERT_VECTOR_EQ(typed_lin_sys.get_b(), additional_rhs);

        ASSERT_MATRIX_EQ(
            typed_lin_sys.get_A_typed(),
            A.template cast<TPrecision>()
        );
        ASSERT_VECTOR_EQ(
            typed_lin_sys.get_b_typed(),
            additional_rhs.template cast<TPrecision>()
        );

    }


    template <template <typename> typename TMatrix, typename TPrecision>
    void TestTypedMutAddlRHSSetRHS() {

        constexpr int m(63);
        constexpr int n(27);
        TMatrix<double> A(
            CommonMatRandomInterface<TMatrix, double>::rand_matrix(
                TestBase::bundle, m, n
            )
        );
        Vector<double> b(
            Vector<double>::Random(TestBase::bundle, m)
        );
        Vector<double> additional_rhs(
            Vector<double>::Random(TestBase::bundle, m)
        );
        Vector<double> new_rhs(
            Vector<double>::Random(TestBase::bundle, m)
        );

        GenericLinearSystem<TMatrix> gen_lin_sys(A, b);
        TypedLinearSystem<TMatrix, TPrecision> temp(&gen_lin_sys);
        TypedLinearSystem_MutAddlRHS typed_lin_sys(&temp, additional_rhs);
        
        // Check b has changed
        typed_lin_sys.set_rhs(new_rhs);
        ASSERT_VECTOR_EQ(typed_lin_sys.get_b(), new_rhs);
        ASSERT_VECTOR_EQ(
            typed_lin_sys.get_b_typed(),
            new_rhs.template cast<TPrecision>()
        );

        // Check remaining values are unchanged
        EXPECT_EQ(typed_lin_sys.get_m(), m);
        EXPECT_EQ(typed_lin_sys.get_n(), n);
        EXPECT_EQ(typed_lin_sys.get_nnz(), A.non_zeros());
        EXPECT_EQ(typed_lin_sys.get_cu_handles(), TestBase::bundle);
        ASSERT_MATRIX_EQ(typed_lin_sys.get_A(), A);
        ASSERT_MATRIX_EQ(
            typed_lin_sys.get_A_typed(),
            A.template cast<TPrecision>()
        );

    }

    template <template <typename> typename TMatrix, typename TPrecision>
    void TestTypedMutAddlRHSBadMismatchSetRHS() {
        
        auto mismatch_setrhs_under = [=]() {

            constexpr int m(63);
            constexpr int n(27);
            TMatrix<double> A(
                CommonMatRandomInterface<TMatrix, double>::rand_matrix(
                    TestBase::bundle, m, n
                )
            );
            Vector<double> b(
                Vector<double>::Random(TestBase::bundle, m)
            );
            Vector<double> additional_rhs(
                Vector<double>::Random(TestBase::bundle, m)
            );

            GenericLinearSystem<TMatrix> gen_lin_sys(A, b);
            TypedLinearSystem<TMatrix, TPrecision> temp(&gen_lin_sys);
            TypedLinearSystem_MutAddlRHS typed_lin_sys(&temp, additional_rhs);

            Vector<double> bad_rhs(
                Vector<double>::Random(TestBase::bundle, m-1)
            );
            typed_lin_sys.set_rhs(bad_rhs);

        };
        
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, mismatch_setrhs_under);
        
        auto mismatch_setrhs_over = [=]() {

            constexpr int m(63);
            constexpr int n(27);
            TMatrix<double> A(
                CommonMatRandomInterface<TMatrix, double>::rand_matrix(
                    TestBase::bundle, m, n
                )
            );
            Vector<double> b(
                Vector<double>::Random(TestBase::bundle, m)
            );
            Vector<double> additional_rhs(
                Vector<double>::Random(TestBase::bundle, m)
            );

            GenericLinearSystem<TMatrix> gen_lin_sys(A, b);
            TypedLinearSystem<TMatrix, TPrecision> temp(&gen_lin_sys);
            TypedLinearSystem_MutAddlRHS typed_lin_sys(&temp, additional_rhs);

            Vector<double> bad_rhs(
                Vector<double>::Random(TestBase::bundle, m+1)
            );
            typed_lin_sys.set_rhs(bad_rhs);

        };
        
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, mismatch_setrhs_over);

    }

};

TEST_F(TypedLinearSystem_Test, TestConstructor_Half) {
    TestTypedConstructor<MatrixDense, __half>();
    TestTypedConstructor<NoFillMatrixSparse, __half>();
}

TEST_F(TypedLinearSystem_Test, TestConstructor_Single) {
    TestTypedConstructor<MatrixDense, float>();
    TestTypedConstructor<NoFillMatrixSparse, float>();
}

TEST_F(TypedLinearSystem_Test, TestConstructor_Double) {
    TestTypedConstructor<MatrixDense, double>();
    TestTypedConstructor<NoFillMatrixSparse, double>();
}

TEST_F(TypedLinearSystem_Test, TestTypedMutAddlRHSConstructor_Half) {
    TestTypedMutAddlRHSConstructor<MatrixDense, __half>();
    TestTypedMutAddlRHSConstructor<NoFillMatrixSparse, __half>();
}

TEST_F(TypedLinearSystem_Test, TestTypedMutAddlRHSConstructor_Single) {
    TestTypedMutAddlRHSConstructor<MatrixDense, float>();
    TestTypedMutAddlRHSConstructor<NoFillMatrixSparse, float>();
}

TEST_F(TypedLinearSystem_Test, TestTypedMutAddlRHSConstructor_Double) {
    TestTypedMutAddlRHSConstructor<MatrixDense, double>();
    TestTypedMutAddlRHSConstructor<NoFillMatrixSparse, double>();
}

TEST_F(TypedLinearSystem_Test, TestMutAddlRHSSetRHS_Half) {
    TestTypedMutAddlRHSSetRHS<MatrixDense, __half>();
    TestTypedMutAddlRHSSetRHS<NoFillMatrixSparse, __half>();
}

TEST_F(TypedLinearSystem_Test, TestMutAddlRHSSetRHS_Single) {
    TestTypedMutAddlRHSSetRHS<MatrixDense, float>();
    TestTypedMutAddlRHSSetRHS<NoFillMatrixSparse, float>();
}

TEST_F(TypedLinearSystem_Test, TestMutAddlRHSSetRHS_Double) {
    TestTypedMutAddlRHSSetRHS<MatrixDense, double>();
    TestTypedMutAddlRHSSetRHS<NoFillMatrixSparse, double>();
}

TEST_F(TypedLinearSystem_Test, TestTypedMutAddlRHSBadMismatchSetRHS_Half) {
    TestTypedMutAddlRHSBadMismatchSetRHS<MatrixDense, __half>();
    TestTypedMutAddlRHSBadMismatchSetRHS<NoFillMatrixSparse, __half>();
}

TEST_F(TypedLinearSystem_Test, TestTypedMutAddlRHSBadMismatchSetRHS_Single) {
    TestTypedMutAddlRHSBadMismatchSetRHS<MatrixDense, float>();
    TestTypedMutAddlRHSBadMismatchSetRHS<NoFillMatrixSparse, float>();
}

TEST_F(TypedLinearSystem_Test, TestTypedMutAddlRHSBadMismatchSetRHS_Double) {
    TestTypedMutAddlRHSBadMismatchSetRHS<MatrixDense, double>();
    TestTypedMutAddlRHSBadMismatchSetRHS<NoFillMatrixSparse, double>();
}