#include "test_Matrix.h"

#include "types/MatrixDense.h"

class MatrixDense_Test: public Matrix_Test
{
public:

    template <typename T>
    void TestCoeffAccess() { TestCoeffAccess_Base<MatrixDense, T>(); }

    void TestBadCoeffAccess() { TestBadCoeffAccess_Base<MatrixDense>(); }

    template <typename T>
    void TestPropertyAccess() { TestPropertyAccess_Base<MatrixDense, T>(); }

    template <typename T>
    void TestConstruction() { TestConstruction_Base<MatrixDense, T>(); }

    template <typename T>
    void TestListInitialization() { TestListInitialization_Base<MatrixDense, T>(); }

    void TestBadListInitialization() { TestBadListInitialization_Base<MatrixDense>(); }

    template <typename T>
    void TestCopyAssignment() { TestCopyAssignment_Base<MatrixDense, T>(); }

    template <typename T>
    void TestCopyConstructor() { TestCopyConstructor_Base<MatrixDense, T>(); }

    template <typename T>
    void TestStaticCreation() { TestStaticCreation_Base<MatrixDense, T>(); }

    template <typename T>
    void TestCol() { TestCol_Base<MatrixDense, T>(); }

    void TestBadCol() { TestBadCol_Base<MatrixDense>(); }

    template <typename T>
    void TestBlock() {

        const MatrixDense<T> const_mat (
            *handle_ptr,
            {{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3),
              static_cast<T>(4), static_cast<T>(5)},
             {static_cast<T>(6), static_cast<T>(7), static_cast<T>(8),
              static_cast<T>(9), static_cast<T>(10)},
             {static_cast<T>(11), static_cast<T>(12), static_cast<T>(13),
              static_cast<T>(14), static_cast<T>(15)},
             {static_cast<T>(16), static_cast<T>(17), static_cast<T>(18),
              static_cast<T>(19), static_cast<T>(20)}}
        );
        MatrixDense<T> mat(const_mat);

        // Test copy constructor and access for block 0, 0, 4, 2
        typename MatrixDense<T>::Block blk_0_0_4_2(mat.get_block(0, 0, 4, 2));
        ASSERT_EQ(blk_0_0_4_2.get_elem(0, 0), static_cast<T>(1));
        ASSERT_EQ(blk_0_0_4_2.get_elem(1, 0), static_cast<T>(6));
        ASSERT_EQ(blk_0_0_4_2.get_elem(2, 0), static_cast<T>(11));
        ASSERT_EQ(blk_0_0_4_2.get_elem(3, 0), static_cast<T>(16));
        ASSERT_EQ(blk_0_0_4_2.get_elem(0, 1), static_cast<T>(2));
        ASSERT_EQ(blk_0_0_4_2.get_elem(1, 1), static_cast<T>(7));
        ASSERT_EQ(blk_0_0_4_2.get_elem(2, 1), static_cast<T>(12));
        ASSERT_EQ(blk_0_0_4_2.get_elem(3, 1), static_cast<T>(17));

        // Test copy constructor and access for block 2, 1, 2, 3
        typename MatrixDense<T>::Block blk_2_1_2_3(mat.get_block(2, 1, 2, 3));
        ASSERT_EQ(blk_2_1_2_3.get_elem(0, 0), static_cast<T>(12));
        ASSERT_EQ(blk_2_1_2_3.get_elem(0, 1), static_cast<T>(13));
        ASSERT_EQ(blk_2_1_2_3.get_elem(0, 2), static_cast<T>(14));
        ASSERT_EQ(blk_2_1_2_3.get_elem(1, 0), static_cast<T>(17));
        ASSERT_EQ(blk_2_1_2_3.get_elem(1, 1), static_cast<T>(18));
        ASSERT_EQ(blk_2_1_2_3.get_elem(1, 2), static_cast<T>(19));

        // Test MatrixDense cast/access for block 0, 0, 3, 4
        MatrixDense<T> mat_0_0_3_4(mat.get_block(0, 0, 3, 4).copy_to_mat());
        MatrixDense<T> test_0_0_3_4(
            *handle_ptr,
            {{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4)},
             {static_cast<T>(6), static_cast<T>(7), static_cast<T>(8), static_cast<T>(9)},
             {static_cast<T>(11), static_cast<T>(12), static_cast<T>(13), static_cast<T>(14)}}
        );
        ASSERT_MATRIX_EQ(mat_0_0_3_4, test_0_0_3_4);

        // Test MatrixDense cast/access for block 1, 2, 3, 1
        MatrixDense<T> mat_1_2_3_1(mat.get_block(1, 2, 3, 1).copy_to_mat());
        MatrixDense<T> test_1_2_3_1(
            *handle_ptr,
            {{static_cast<T>(8)},
             {static_cast<T>(13)},
             {static_cast<T>(18)}}
        );
        ASSERT_MATRIX_EQ(mat_1_2_3_1, test_1_2_3_1);

        // Test assignment from MatrixDense
        mat = const_mat;
        MatrixDense<T> zero_2_3(MatrixDense<T>::Zero(*handle_ptr, 2, 3));
        mat.get_block(1, 1, 2, 3).set_from_mat(zero_2_3);
        MatrixDense<T> test_assign_2_3(
            *handle_ptr,
            {{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3),
              static_cast<T>(4), static_cast<T>(5)},
             {static_cast<T>(6), static_cast<T>(0), static_cast<T>(0),
              static_cast<T>(0), static_cast<T>(10)},
             {static_cast<T>(11), static_cast<T>(0), static_cast<T>(0),
              static_cast<T>(0), static_cast<T>(15)},
             {static_cast<T>(16), static_cast<T>(17), static_cast<T>(18),
              static_cast<T>(19), static_cast<T>(20)}}
        );
        ASSERT_MATRIX_EQ(mat, test_assign_2_3);

        // Test assignment from MatrixVector
        mat = const_mat;
        MatrixVector<T> assign_vec(
            *handle_ptr,
            {static_cast<T>(1),
             static_cast<T>(1),
             static_cast<T>(1)}
        );
        mat.get_block(1, 4, 3, 1).set_from_vec(assign_vec);
        MatrixDense<T> test_assign_1_4(
            *handle_ptr,
            {{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3),
              static_cast<T>(4), static_cast<T>(5)},
             {static_cast<T>(6), static_cast<T>(7), static_cast<T>(8),
              static_cast<T>(9), static_cast<T>(1)},
             {static_cast<T>(11), static_cast<T>(12), static_cast<T>(13),
              static_cast<T>(14), static_cast<T>(1)},
             {static_cast<T>(16), static_cast<T>(17), static_cast<T>(18),
              static_cast<T>(19), static_cast<T>(1)}}
        );
        ASSERT_MATRIX_EQ(mat, test_assign_1_4);

    }

    void TestBadBlock() {

        const int m(4);
        const int n(5);
        const MatrixDense<double> const_mat (
            *handle_ptr,
            {{1, 2, 3, 4, 5},
             {6, 7, 8, 9, 10},
             {11, 12, 13, 14, 15},
             {16, 17, 18, 19, 20}}
        );
        MatrixDense<double> mat(const_mat);

        // Test invalid starts
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() mutable { mat.get_block(-1, 0, 1, 1); });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() mutable { mat.get_block(m, 0, 1, 1); });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() mutable { mat.get_block(0, -1, 1, 1); });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() mutable { mat.get_block(0, n, 1, 1); });

        // Test invalid sizes from 0
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() mutable { mat.get_block(0, 0, -1, 1); });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() mutable { mat.get_block(0, 0, 1, -1); });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() mutable { mat.get_block(0, 0, m+1, 1); });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() mutable { mat.get_block(0, 0, 1, n+1); });

        // Test invalid sizes from not initial index
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() mutable { mat.get_block(1, 2, -1, 1); });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() mutable { mat.get_block(1, 2, 1, -1); });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() mutable { mat.get_block(1, 2, m, 1); });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() mutable { mat.get_block(1, 2, 1, n-1); });

        // Test invalid access to valid block
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() mutable { mat.get_block(1, 2, 2, 2).get_elem(-1, 0); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() mutable { mat.get_block(1, 2, 2, 2).get_elem(0, -1); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() mutable { mat.get_block(1, 2, 2, 2).get_elem(2, 0); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() mutable { mat.get_block(1, 2, 2, 2).get_elem(0, 2); }
        );

    }

    template <typename T>
    void TestScale() { TestScale_Base<MatrixDense, T>(); }

    template <typename T>
    void TestMatVec() { TestMatVec_Base<MatrixDense, T>(); }

    template <typename T>
    void TestBadMatVec() { TestBadMatVec_Base<MatrixDense, T>(); }

    template <typename T>
    void TestTransposeMatVec() { TestTransposeMatVec_Base<MatrixDense, T>(); }

    template <typename T>
    void TestBadTransposeMatVec() { TestBadTransposeMatVec_Base<MatrixDense, T>(); }

    template <typename T>
    void TestTranspose() { TestTranspose_Base<MatrixDense, T>(); }

    template <typename T>
    void TestMatMat() { TestMatMat_Base<MatrixDense, T>(); }

    template <typename T>
    void TestBadMatMat() { TestBadMatMat_Base<MatrixDense, T>(); }

    template <typename T>
    void TestAddSub() { TestAddSub_Base<MatrixDense, T>(); }

    template <typename T>
    void TestBadAddSub() { TestBadAddSub_Base<MatrixDense, T>(); }

    template <typename T>
    void TestNorm() { TestNorm_Base<MatrixDense, T>(); }

    void TestCast() { TestCast_Base<MatrixDense>(); }

};

TEST_F(MatrixDense_Test, TestCoeffAccess) {
    TestCoeffAccess<half>();
    TestCoeffAccess<float>();
    TestCoeffAccess<double>();
}

TEST_F(MatrixDense_Test, TestBadCoeffAccess) { TestBadCoeffAccess(); }

TEST_F(MatrixDense_Test, TestPropertyAccess) {
    TestPropertyAccess<half>();
    TestPropertyAccess<float>();
    TestPropertyAccess<double>();
}

TEST_F(MatrixDense_Test, TestConstruction) {
    TestConstruction<half>();
    TestConstruction<float>();
    TestConstruction<double>();
}

TEST_F(MatrixDense_Test, TestListInitialization) {
    TestListInitialization<half>();
    TestListInitialization<float>();
    TestListInitialization<double>();
}

TEST_F(MatrixDense_Test, TestBadListInitialization) { TestBadListInitialization(); }

TEST_F(MatrixDense_Test, TestCopyAssignment) {
    TestCopyAssignment<half>();
    TestCopyAssignment<float>();
    TestCopyAssignment<double>();
}

TEST_F(MatrixDense_Test, TestCopyConstructor) {
    TestCopyConstructor<half>();
    TestCopyConstructor<float>();
    TestCopyConstructor<double>();
}

TEST_F(MatrixDense_Test, TestStaticCreation) {
    TestStaticCreation<half>();
    TestStaticCreation<float>();
    TestStaticCreation<double>();
}

TEST_F(MatrixDense_Test, TestCol) {
    TestCol<half>();
    TestCol<float>();
    TestCol<double>();
}

TEST_F(MatrixDense_Test, TestBadCol) { TestBadCol(); }

TEST_F(MatrixDense_Test, TestBlock) {
    TestBlock<half>();
    TestBlock<float>();
    TestBlock<double>();
}

TEST_F(MatrixDense_Test, TestBadBlock) { TestBadBlock(); }

TEST_F(MatrixDense_Test, TestScale) {
    TestScale<half>();
    TestScale<float>();
    TestScale<double>();
}

TEST_F(MatrixDense_Test, TestMatVec) {
    TestMatVec<half>();
    TestMatVec<float>();
    TestMatVec<double>();
}

TEST_F(MatrixDense_Test, TestBadMatVec) {
    TestBadMatVec<half>();
    TestBadMatVec<float>();
    TestBadMatVec<double>();
}

TEST_F(MatrixDense_Test, TestTransposeMatVec) {
    TestTransposeMatVec<half>();
    TestTransposeMatVec<float>();
    TestTransposeMatVec<double>();
}

TEST_F(MatrixDense_Test, TestBadTransposeMatVec) {
    TestBadTransposeMatVec<half>();
    TestBadTransposeMatVec<float>();
    TestBadTransposeMatVec<double>();
}

TEST_F(MatrixDense_Test, TestTranspose) {
    TestTranspose<half>();
    TestTranspose<float>();
    TestTranspose<double>();
}

TEST_F(MatrixDense_Test, TestMatMat) {
    TestMatMat<half>();
    TestMatMat<float>();
    TestMatMat<double>();
}

TEST_F(MatrixDense_Test, TestBadMatMat) {
    TestBadMatMat<half>();
    TestBadMatMat<float>();
    TestBadMatMat<double>();
}

TEST_F(MatrixDense_Test, TestAddSub) {
    TestAddSub<half>();
    TestAddSub<float>();
    TestAddSub<double>();
}

TEST_F(MatrixDense_Test, TestBadAddSub) {
    TestBadAddSub<half>();
    TestBadAddSub<float>();
    TestBadAddSub<double>();
}

TEST_F(MatrixDense_Test, TestNorm) {
    TestNorm<half>();
    TestNorm<float>();
    TestNorm<double>();
}

TEST_F(MatrixDense_Test, TestCast) { TestCast(); }

class MatrixDense_Substitution_Test: public TestBase
{
public:

    template <template <typename> typename M>
    void TestBackwardSubstitution() {

        constexpr int n(90);
        M<double> U_tri(
            read_matrixCSV<M, double>(*handle_ptr, solve_matrix_dir / fs::path("U_tri_90.csv"))
        );
        MatrixVector<double> x_tri(
            read_matrixCSV<MatrixVector, double>(*handle_ptr, solve_matrix_dir / fs::path("x_tri_90.csv"))
        );
        MatrixVector<double> Ub_tri(
            read_matrixCSV<MatrixVector, double>(*handle_ptr, solve_matrix_dir / fs::path("Ub_tri_90.csv"))
        );
    
        MatrixVector<double> test_soln(U_tri.back_sub(Ub_tri));

        ASSERT_VECTOR_NEAR(test_soln, x_tri, Tol<double>::dbl_substitution_tol());

    }

    template <template <typename> typename M>
    void TestForwardSubstitution() {

        constexpr int n(90);
        M<double> L_tri(
            read_matrixCSV<M, double>(*handle_ptr, solve_matrix_dir / fs::path("L_tri_90.csv"))
        );
        MatrixVector<double> x_tri(
            read_matrixCSV<MatrixVector, double>(*handle_ptr, solve_matrix_dir / fs::path("x_tri_90.csv"))
        );
        MatrixVector<double> Lb_tri(
            read_matrixCSV<MatrixVector, double>(*handle_ptr, solve_matrix_dir / fs::path("Lb_tri_90.csv"))
        );
    
        MatrixVector<double> test_soln(L_tri.frwd_sub(Lb_tri));

        ASSERT_VECTOR_NEAR(test_soln, x_tri, Tol<double>::dbl_substitution_tol());

    }

};

TEST_F(MatrixDense_Substitution_Test, TestBackwardSubstitution) {
    TestBackwardSubstitution<MatrixDense>();
    // TestBackwardSubstitution<MatrixSparse>();
}

TEST_F(MatrixDense_Substitution_Test, TestForwardSubstitution) {
    TestForwardSubstitution<MatrixDense>();
//     // TestForwardSubstitution<MatrixSparse>();
}