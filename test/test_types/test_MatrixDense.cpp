#include "test_Matrix.h"

#include "types/MatrixDense/MatrixDense.h"

class MatrixDense_Test: public Matrix_Test<MatrixDense>
{
public:

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
        ASSERT_EQ(blk_0_0_4_2.get_elem(0, 0).get_scalar(), static_cast<T>(1));
        ASSERT_EQ(blk_0_0_4_2.get_elem(1, 0).get_scalar(), static_cast<T>(6));
        ASSERT_EQ(blk_0_0_4_2.get_elem(2, 0).get_scalar(), static_cast<T>(11));
        ASSERT_EQ(blk_0_0_4_2.get_elem(3, 0).get_scalar(), static_cast<T>(16));
        ASSERT_EQ(blk_0_0_4_2.get_elem(0, 1).get_scalar(), static_cast<T>(2));
        ASSERT_EQ(blk_0_0_4_2.get_elem(1, 1).get_scalar(), static_cast<T>(7));
        ASSERT_EQ(blk_0_0_4_2.get_elem(2, 1).get_scalar(), static_cast<T>(12));
        ASSERT_EQ(blk_0_0_4_2.get_elem(3, 1).get_scalar(), static_cast<T>(17));

        // Test copy constructor and access for block 2, 1, 2, 3
        typename MatrixDense<T>::Block blk_2_1_2_3(mat.get_block(2, 1, 2, 3));
        ASSERT_EQ(blk_2_1_2_3.get_elem(0, 0).get_scalar(), static_cast<T>(12));
        ASSERT_EQ(blk_2_1_2_3.get_elem(0, 1).get_scalar(), static_cast<T>(13));
        ASSERT_EQ(blk_2_1_2_3.get_elem(0, 2).get_scalar(), static_cast<T>(14));
        ASSERT_EQ(blk_2_1_2_3.get_elem(1, 0).get_scalar(), static_cast<T>(17));
        ASSERT_EQ(blk_2_1_2_3.get_elem(1, 1).get_scalar(), static_cast<T>(18));
        ASSERT_EQ(blk_2_1_2_3.get_elem(1, 2).get_scalar(), static_cast<T>(19));

        // Test MatrixDense cast/access for block 0, 0, 3, 4
        MatrixDense<T> mat_0_0_3_4(mat.get_block(0, 0, 3, 4));
        MatrixDense<T> test_0_0_3_4(
            *handle_ptr,
            {{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4)},
             {static_cast<T>(6), static_cast<T>(7), static_cast<T>(8), static_cast<T>(9)},
             {static_cast<T>(11), static_cast<T>(12), static_cast<T>(13), static_cast<T>(14)}}
        );
        ASSERT_MATRIX_EQ(mat_0_0_3_4, test_0_0_3_4);

        // Test MatrixDense cast/access for block 1, 2, 3, 1
        MatrixDense<T> mat_1_2_3_1(mat.get_block(1, 2, 3, 1));
        MatrixDense<T> test_1_2_3_1(
            *handle_ptr,
            {{static_cast<T>(8)},
             {static_cast<T>(13)},
             {static_cast<T>(18)}}
        );
        ASSERT_MATRIX_EQ(mat_1_2_3_1, test_1_2_3_1);

        // Test MatrixDense cast/access for block 0, 0, 3, 4
        MatrixDense<T> mat_0_0_3_4_direct(mat.get_block(0, 0, 3, 4));
        ASSERT_MATRIX_EQ(mat_0_0_3_4_direct, test_0_0_3_4);

        // Test MatrixDense cast/access for block 1, 2, 3, 1
        MatrixDense<T> mat_1_2_3_1_direct(mat.get_block(1, 2, 3, 1));
        ASSERT_MATRIX_EQ(mat_1_2_3_1_direct, test_1_2_3_1);

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

        // Test assignment from Vector
        mat = const_mat;
        Vector<T> assign_vec(
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

};

TEST_F(MatrixDense_Test, TestCoeffAccess) {
    TestCoeffAccess<__half>();
    TestCoeffAccess<float>();
    TestCoeffAccess<double>();
}

TEST_F(MatrixDense_Test, TestBadCoeffAccess) { TestBadCoeffAccess(); }

TEST_F(MatrixDense_Test, TestPropertyAccess) {
    TestPropertyAccess<__half>();
    TestPropertyAccess<float>();
    TestPropertyAccess<double>();
}

TEST_F(MatrixDense_Test, TestConstruction) {
    TestConstruction<__half>();
    TestConstruction<float>();
    TestConstruction<double>();
}

TEST_F(MatrixDense_Test, TestListInitialization) {
    TestListInitialization<__half>();
    TestListInitialization<float>();
    TestListInitialization<double>();
}

TEST_F(MatrixDense_Test, TestBadListInitialization) { TestBadListInitialization(); }

TEST_F(MatrixDense_Test, TestNonZeros) {
    TestNonZeros<__half>();
    TestNonZeros<float>();
    TestNonZeros<double>();
}

TEST_F(MatrixDense_Test, TestDynamicMemConstruction) {
    TestDynamicMemConstruction<__half>();
    TestDynamicMemConstruction<float>();
    TestDynamicMemConstruction<double>();
}

TEST_F(MatrixDense_Test, TestDynamicMemCopyToPtr) {
    TestDynamicMemCopyToPtr<__half>();
    TestDynamicMemCopyToPtr<float>();
    TestDynamicMemCopyToPtr<double>();
}

TEST_F(MatrixDense_Test, TestBadDynamicMemCopyToPtr) { TestBadDynamicMemCopyToPtr(); }

TEST_F(MatrixDense_Test, TestCopyAssignment) {
    TestCopyAssignment<__half>();
    TestCopyAssignment<float>();
    TestCopyAssignment<double>();
}

TEST_F(MatrixDense_Test, TestCopyConstructor) {
    TestCopyConstructor<__half>();
    TestCopyConstructor<float>();
    TestCopyConstructor<double>();
}

TEST_F(MatrixDense_Test, TestStaticCreation) {
    TestStaticCreation<__half>();
    TestStaticCreation<float>();
    TestStaticCreation<double>();
}

TEST_F(MatrixDense_Test, TestCol) {
    TestCol<__half>();
    TestCol<float>();
    TestCol<double>();
}

TEST_F(MatrixDense_Test, TestBadCol) { TestBadCol(); }

TEST_F(MatrixDense_Test, TestBlock) {
    TestBlock<__half>();
    TestBlock<float>();
    TestBlock<double>();
}

TEST_F(MatrixDense_Test, TestBadBlock) { TestBadBlock(); }

TEST_F(MatrixDense_Test, TestScale) {
    TestScale<__half>();
    TestScale<float>();
    TestScale<double>();
}

TEST_F(MatrixDense_Test, TestScaleAssignment) {
    TestScaleAssignment<__half>();
    TestScaleAssignment<float>();
    TestScaleAssignment<double>();
}

TEST_F(MatrixDense_Test, TestMaxMagElem) {
    TestMaxMagElem<__half>();
    TestMaxMagElem<float>();
    TestMaxMagElem<double>();
}

TEST_F(MatrixDense_Test, TestNormalizeMagnitude) {
    TestNormalizeMagnitude<__half>();
    TestNormalizeMagnitude<float>();
    TestNormalizeMagnitude<double>();
}

TEST_F(MatrixDense_Test, TestMatVec) {
    TestMatVec<__half>();
    TestMatVec<float>();
    TestMatVec<double>();
}

TEST_F(MatrixDense_Test, TestBadMatVec) {
    TestBadMatVec<__half>();
    TestBadMatVec<float>();
    TestBadMatVec<double>();
}

TEST_F(MatrixDense_Test, TestTransposeMatVec) {
    TestTransposeMatVec<__half>();
    TestTransposeMatVec<float>();
    TestTransposeMatVec<double>();
}

TEST_F(MatrixDense_Test, TestBadTransposeMatVec) {
    TestBadTransposeMatVec<__half>();
    TestBadTransposeMatVec<float>();
    TestBadTransposeMatVec<double>();
}

TEST_F(MatrixDense_Test, TestTranspose) {
    TestTranspose<__half>();
    TestTranspose<float>();
    TestTranspose<double>();
}

TEST_F(MatrixDense_Test, TestMatMat) {
    TestMatMat<__half>();
    TestMatMat<float>();
    TestMatMat<double>();
}

TEST_F(MatrixDense_Test, TestBadMatMat) {
    TestBadMatMat<__half>();
    TestBadMatMat<float>();
    TestBadMatMat<double>();
}

TEST_F(MatrixDense_Test, TestAddSub) {
    TestAddSub<__half>();
    TestAddSub<float>();
    TestAddSub<double>();
}

TEST_F(MatrixDense_Test, TestBadAddSub) {
    TestBadAddSub<__half>();
    TestBadAddSub<float>();
    TestBadAddSub<double>();
}

TEST_F(MatrixDense_Test, TestNorm) {
    TestNorm<__half>();
    TestNorm<float>();
    TestNorm<double>();
}

TEST_F(MatrixDense_Test, TestCast) { TestCast(); }

TEST_F(MatrixDense_Test, TestBadCast) { TestBadCast(); }

class MatrixDense_Substitution_Test: public Matrix_Substitution_Test<MatrixDense> {};

TEST_F(MatrixDense_Substitution_Test, TestBackwardSubstitution) {
    TestBackwardSubstitution<__half>();
    TestBackwardSubstitution<float>();
    TestBackwardSubstitution<double>();
}

TEST_F(MatrixDense_Substitution_Test, TestForwardSubstitution) {
    TestForwardSubstitution<__half>();
    TestForwardSubstitution<float>();
    TestForwardSubstitution<double>();
}