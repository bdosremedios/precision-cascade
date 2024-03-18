#include "test_Matrix.h"

#include "types/MatrixSparse/ImmutableMatrixSparse.h"

class ImmutableMatrixSparse_Test: public Matrix_Test<ImmutableMatrixSparse>
{
public:

    template <typename T>
    void TestCoeffJustGetAccess() {

        const int m(5);
        const int n(4);

        // Test simple counting up
        ImmutableMatrixSparse<T> test_mat_1(
            TestBase::bundle,
            {{static_cast<T>(1.), static_cast<T>(2.), static_cast<T>(3.), static_cast<T>(4.)},
             {static_cast<T>(5.), static_cast<T>(6.), static_cast<T>(7.), static_cast<T>(8.)},
             {static_cast<T>(9.), static_cast<T>(10.), static_cast<T>(11.), static_cast<T>(12.)},
             {static_cast<T>(13.), static_cast<T>(14.), static_cast<T>(15.), static_cast<T>(16.)},
             {static_cast<T>(17.), static_cast<T>(18.), static_cast<T>(19.), static_cast<T>(20.)}}
        );

        T test_elem = static_cast<T>(1.);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_EQ(test_mat_1.get_elem(i, j).get_scalar(), test_elem);
                test_elem += static_cast<T>(1);
            }
        }

        // Modify first row to be all -1.
        ImmutableMatrixSparse<T> test_mat_2(
            TestBase::bundle,
            {{static_cast<T>(-1.), static_cast<T>(-1.), static_cast<T>(-1.), static_cast<T>(-1.)},
             {static_cast<T>(5.), static_cast<T>(6.), static_cast<T>(7.), static_cast<T>(8.)},
             {static_cast<T>(9.), static_cast<T>(10.), static_cast<T>(11.), static_cast<T>(12.)},
             {static_cast<T>(13.), static_cast<T>(14.), static_cast<T>(15.), static_cast<T>(16.)},
             {static_cast<T>(17.), static_cast<T>(18.), static_cast<T>(19.), static_cast<T>(20.)}}
        );

        test_elem = static_cast<T>(1);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                if (i == 0) {
                    ASSERT_EQ(test_mat_2.get_elem(i, j).get_scalar(), static_cast<T>(-1.));
                } else {
                    ASSERT_EQ(test_mat_2.get_elem(i, j).get_scalar(), test_elem);
                }
                test_elem += static_cast<T>(1);
            }
        }

        // Set index 4 row as decreasing by -1 from -1
        ImmutableMatrixSparse<T> test_mat_3(
            TestBase::bundle,
            {{static_cast<T>(-1.), static_cast<T>(-1.), static_cast<T>(-1.), static_cast<T>(-1.)},
             {static_cast<T>(5.), static_cast<T>(6.), static_cast<T>(7.), static_cast<T>(8.)},
             {static_cast<T>(9.), static_cast<T>(10.), static_cast<T>(11.), static_cast<T>(12.)},
             {static_cast<T>(13.), static_cast<T>(14.), static_cast<T>(15.), static_cast<T>(16.)},
             {static_cast<T>(-1.), static_cast<T>(-2.), static_cast<T>(-3.), static_cast<T>(-4.)}}
        );

        test_elem = static_cast<T>(1);
        T row_5_test_elem = static_cast<T>(-1.);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                if (i == 0) {
                    ASSERT_EQ(test_mat_3.get_elem(i, j).get_scalar(), static_cast<T>(-1.));
                } else if (i == 4) {
                    ASSERT_EQ(test_mat_3.get_elem(i, j).get_scalar(), row_5_test_elem);
                    row_5_test_elem += static_cast<T>(-1.);
                } else {
                    ASSERT_EQ(test_mat_3.get_elem(i, j).get_scalar(), test_elem);
                }
                test_elem += static_cast<T>(1);
            }
        }

        // Set index 2 col as increasing by 1 from -5
        ImmutableMatrixSparse<T> test_mat_4(
            TestBase::bundle,
            {{static_cast<T>(-1.), static_cast<T>(-1.), static_cast<T>(-5.), static_cast<T>(-1.)},
             {static_cast<T>(5.), static_cast<T>(6.), static_cast<T>(-4.), static_cast<T>(8.)},
             {static_cast<T>(9.), static_cast<T>(10.), static_cast<T>(-3.), static_cast<T>(12.)},
             {static_cast<T>(13.), static_cast<T>(14.), static_cast<T>(-2.), static_cast<T>(16.)},
             {static_cast<T>(-1.), static_cast<T>(-2.), static_cast<T>(-1.), static_cast<T>(-4.)}}
        );

        test_elem = static_cast<T>(1);
        row_5_test_elem = static_cast<T>(-1.);
        T coL_3_test_elem = static_cast<T>(-5.);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                if (j == 2) {
                    ASSERT_EQ(test_mat_4.get_elem(i, j).get_scalar(), coL_3_test_elem);
                    coL_3_test_elem += static_cast<T>(1.);
                } else if (i == 0) {
                    ASSERT_EQ(test_mat_4.get_elem(i, j).get_scalar(), static_cast<T>(-1.));
                } else if (i == 4) {
                    ASSERT_EQ(test_mat_4.get_elem(i, j).get_scalar(), row_5_test_elem);
                    row_5_test_elem += static_cast<T>(-1.);
                    if (j == 1) { row_5_test_elem += static_cast<T>(-1.); }
                } else {
                    ASSERT_EQ(test_mat_4.get_elem(i, j).get_scalar(), test_elem);
                }
                test_elem += static_cast<T>(1);
            }
        }

        // Test reading an all zero matrix
        ImmutableMatrixSparse<T> test_mat_5(
            TestBase::bundle,
            {{static_cast<T>(0.), static_cast<T>(0.), static_cast<T>(0.), static_cast<T>(0.)},
             {static_cast<T>(0.), static_cast<T>(0.), static_cast<T>(0.), static_cast<T>(0.)},
             {static_cast<T>(0.), static_cast<T>(0.), static_cast<T>(0.), static_cast<T>(0.)},
             {static_cast<T>(0.), static_cast<T>(0.), static_cast<T>(0.), static_cast<T>(0.)},
             {static_cast<T>(0.), static_cast<T>(0.), static_cast<T>(0.), static_cast<T>(0.)}}
        );
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_EQ(test_mat_5.get_elem(i, j).get_scalar(), static_cast<T>(0.));
            }
        }

        // Test reading a sparse matrix
        ImmutableMatrixSparse<T> test_mat_6(
            TestBase::bundle,
            {{static_cast<T>(1.), static_cast<T>(0.), static_cast<T>(0.), static_cast<T>(0.)},
             {static_cast<T>(0.), static_cast<T>(0.), static_cast<T>(0.), static_cast<T>(4.)},
             {static_cast<T>(0.), static_cast<T>(0.), static_cast<T>(0.), static_cast<T>(0.)},
             {static_cast<T>(0.), static_cast<T>(-5.), static_cast<T>(0.), static_cast<T>(0.)},
             {static_cast<T>(0.), static_cast<T>(0.), static_cast<T>(0.), static_cast<T>(0.)}}
        );
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                if ((i == 0) && (j == 0)) {
                    ASSERT_EQ(test_mat_6.get_elem(i, j).get_scalar(), static_cast<T>(1.));
                } else if ((i == 1) && (j == 3)) {
                    ASSERT_EQ(test_mat_6.get_elem(i, j).get_scalar(), static_cast<T>(4.));
                } else if ((i == 3) && (j == 1)) {
                    ASSERT_EQ(test_mat_6.get_elem(i, j).get_scalar(), static_cast<T>(-5.));
                } else {
                    ASSERT_EQ(test_mat_6.get_elem(i, j).get_scalar(), static_cast<T>(0.));
                }
            }
        }
    
    }

    void TestBadCoeffJustGetAccess() {
        
        constexpr int m(24);
        constexpr int n(12);
        ImmutableMatrixSparse<double> test_mat(TestBase::bundle, m, n);
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [&]() { test_mat.get_elem(0, -1); });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [&]() { test_mat.get_elem(0, n); });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [&]() { test_mat.get_elem(-1, 0); });
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [&]() { test_mat.get_elem(m, 0); });

    }

    // template <typename T>
    // void TestCopyAssignment() { TestCopyAssignment_Base<MatrixSparse, T>(); }

    // template <typename T>
    // void TestCopyConstructor() { TestCopyConstructor_Base<MatrixSparse, T>(); }

    // template <typename T>
    // void TestStaticCreation() { TestStaticCreation_Base<MatrixSparse, T>(); }

    // template <typename T>
    // void TestCol() { TestCol_Base<MatrixSparse, T>(); }

    // template <typename T>
    // void TestTranspose() { TestTranspose_Base<MatrixSparse, T>(); }

    // template <typename T>
    // void TestScale() { TestScale_Base<MatrixSparse, T>(); }

    // template <typename T>
    // void TestMatVec() { TestMatVec_Base<MatrixSparse, T>(); }

    // template <typename T>
    // void TestMatMat() { TestMatMat_Base<MatrixSparse, T>(); }

    // template <typename T>
    // void TestNorm() { TestNorm_Base<MatrixSparse, T>(); }

    // template <typename T>
    // void TestAddSub() { TestAddSub_Base<MatrixSparse, T>(); }

    // void TestCast() { TestCast_Base<MatrixSparse>(); }

};

TEST_F(ImmutableMatrixSparse_Test, TestPropertyAccess) {
    TestPropertyAccess<__half>();
    TestPropertyAccess<float>();
    TestPropertyAccess<double>();
}

TEST_F(ImmutableMatrixSparse_Test, TestNonZeros) {
    TestNonZeros<__half>();
    TestNonZeros<float>();
    TestNonZeros<double>();
}

TEST_F(ImmutableMatrixSparse_Test, TestConstruction) {
    TestConstruction<__half>();
    TestConstruction<float>();
    TestConstruction<double>();
}

TEST_F(ImmutableMatrixSparse_Test, TestListInitialization) {
    TestListInitialization<__half>();
    TestListInitialization<float>();
    TestListInitialization<double>();
}

TEST_F(ImmutableMatrixSparse_Test, TestBadListInitialization) {
    TestBadListInitialization();
}

TEST_F(ImmutableMatrixSparse_Test, TestCoeffJustGetAccess) {
    TestCoeffJustGetAccess<__half>();
    TestCoeffJustGetAccess<float>();
    TestCoeffJustGetAccess<double>();
}

TEST_F(ImmutableMatrixSparse_Test, TestBadCoeffJustGetAccess) {
    TestBadCoeffJustGetAccess();
}

// TEST_F(MatrixSparse_Test, TestDynamicMemConstruction) {
//     TestDynamicMemConstruction<__half>();
//     TestDynamicMemConstruction<float>();
//     TestDynamicMemConstruction<double>();
// }

// TEST_F(MatrixSparse_Test, TestDynamicMemCopyToPtr) {
//     TestDynamicMemCopyToPtr<__half>();
//     TestDynamicMemCopyToPtr<float>();
//     TestDynamicMemCopyToPtr<double>();
// }

// TEST_F(MatrixSparse_Test, TestBadDynamicMemCopyToPtr) { TestBadDynamicMemCopyToPtr(); }

// TEST_F(MatrixSparse_Test, TestCopyAssignment) {
//     TestCopyAssignment<__half>();
//     TestCopyAssignment<float>();
//     TestCopyAssignment<double>();
// }

// TEST_F(MatrixSparse_Test, TestCopyConstructor) {
//     TestCopyConstructor<__half>();
//     TestCopyConstructor<float>();
//     TestCopyConstructor<double>();
// }

// TEST_F(MatrixSparse_Test, TestStaticCreation) {
//     TestStaticCreation<__half>();
//     TestStaticCreation<float>();
//     TestStaticCreation<double>();
// }

// TEST_F(MatrixSparse_Test, TestCol) {
//     TestCol<__half>();
//     TestCol<float>();
//     TestCol<double>();
// }

// TEST_F(MatrixSparse_Test, TestBadCol) { TestBadCol(); }

// TEST_F(MatrixSparse_Test, TestBlock) {
//     TestBlock<__half>();
//     TestBlock<float>();
//     TestBlock<double>();
// }

// TEST_F(MatrixSparse_Test, TestBadBlock) { TestBadBlock(); }

// TEST_F(MatrixSparse_Test, TestScale) {
//     TestScale<__half>();
//     TestScale<float>();
//     TestScale<double>();
// }

// TEST_F(MatrixSparse_Test, TestScaleAssignment) {
//     TestScaleAssignment<__half>();
//     TestScaleAssignment<float>();
//     TestScaleAssignment<double>();
// }

// TEST_F(MatrixSparse_Test, TestMaxMagElem) {
//     TestMaxMagElem<__half>();
//     TestMaxMagElem<float>();
//     TestMaxMagElem<double>();
// }

// TEST_F(MatrixSparse_Test, TestNormalizeMagnitude) {
//     TestNormalizeMagnitude<__half>();
//     TestNormalizeMagnitude<float>();
//     TestNormalizeMagnitude<double>();
// }

// TEST_F(MatrixSparse_Test, TestMatVec) {
//     TestMatVec<__half>();
//     TestMatVec<float>();
//     TestMatVec<double>();
// }

// TEST_F(MatrixSparse_Test, TestBadMatVec) {
//     TestBadMatVec<__half>();
//     TestBadMatVec<float>();
//     TestBadMatVec<double>();
// }

// TEST_F(MatrixSparse_Test, TestTransposeMatVec) {
//     TestTransposeMatVec<__half>();
//     TestTransposeMatVec<float>();
//     TestTransposeMatVec<double>();
// }

// TEST_F(MatrixSparse_Test, TestBadTransposeMatVec) {
//     TestBadTransposeMatVec<__half>();
//     TestBadTransposeMatVec<float>();
//     TestBadTransposeMatVec<double>();
// }

// TEST_F(MatrixSparse_Test, TestTranspose) {
//     TestTranspose<__half>();
//     TestTranspose<float>();
//     TestTranspose<double>();
// }

// TEST_F(MatrixSparse_Test, TestMatMat) {
//     TestMatMat<__half>();
//     TestMatMat<float>();
//     TestMatMat<double>();
// }

// TEST_F(MatrixSparse_Test, TestBadMatMat) {
//     TestBadMatMat<__half>();
//     TestBadMatMat<float>();
//     TestBadMatMat<double>();
// }

// TEST_F(MatrixSparse_Test, TestAddSub) {
//     TestAddSub<__half>();
//     TestAddSub<float>();
//     TestAddSub<double>();
// }

// TEST_F(MatrixSparse_Test, TestBadAddSub) {
//     TestBadAddSub<__half>();
//     TestBadAddSub<float>();
//     TestBadAddSub<double>();
// }

// TEST_F(MatrixSparse_Test, TestNorm) {
//     TestNorm<__half>();
//     TestNorm<float>();
//     TestNorm<double>();
// }

// TEST_F(MatrixSparse_Test, TestCast) { TestCast(); }

// TEST_F(MatrixSparse_Test, TestBadCast) { TestBadCast(); }

// class MatrixSparse_Substitution_Test: public Matrix_Substitution_Test<MatrixSparse> {};

// TEST_F(MatrixSparse_Substitution_Test, TestBackwardSubstitution) {
//     TestBackwardSubstitution<__half>();
//     TestBackwardSubstitution<float>();
//     TestBackwardSubstitution<double>();
// }

// TEST_F(MatrixSparse_Substitution_Test, TestForwardSubstitution) {
//     TestForwardSubstitution<__half>();
//     TestForwardSubstitution<float>();
//     TestForwardSubstitution<double>();
// }