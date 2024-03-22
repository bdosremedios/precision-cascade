#include "test_Matrix.h"

#include "types/MatrixDense/MatrixDense.h"

class MatrixDense_Test: public Matrix_Test<MatrixDense>
{
public:

    template <typename T>
    void TestDynamicMemConstruction() {
    
        const int m_manual(2);
        const int n_manual(3);
        T *h_mat_manual = static_cast<T *>(malloc(m_manual*n_manual*sizeof(T)));
        h_mat_manual[0+0*m_manual] = static_cast<T>(-5);
        h_mat_manual[1+0*m_manual] = static_cast<T>(3);
        h_mat_manual[0+1*m_manual] = static_cast<T>(100);
        h_mat_manual[1+1*m_manual] = static_cast<T>(3.5);
        h_mat_manual[0+2*m_manual] = static_cast<T>(-20);
        h_mat_manual[1+2*m_manual] = static_cast<T>(3);

        MatrixDense<T> test_mat_manual(TestBase::bundle, h_mat_manual, m_manual, n_manual);

        MatrixDense<T> target_mat_manual(
            TestBase::bundle,
            {{static_cast<T>(-5), static_cast<T>(100), static_cast<T>(-20)},
             {static_cast<T>(3), static_cast<T>(3.5), static_cast<T>(3)}}
        );

        ASSERT_MATRIX_EQ(test_mat_manual, target_mat_manual);

        free(h_mat_manual);
    
        const int m_rand(4);
        const int n_rand(5);
        T *h_mat_rand = static_cast<T *>(malloc(m_rand*n_rand*sizeof(T)));
        for (int i=0; i<m_rand; ++i) {
            for (int j=0; j<n_rand; ++j) {
                h_mat_rand[i+j*m_rand] = static_cast<T>(rand());
            }
        }

        MatrixDense<T> test_mat_rand(TestBase::bundle, h_mat_rand, m_rand, n_rand);

        ASSERT_EQ(test_mat_rand.rows(), m_rand);
        ASSERT_EQ(test_mat_rand.cols(), n_rand);
        for (int i=0; i<m_rand; ++i) {
            for (int j=0; j<n_rand; ++j) {
                ASSERT_EQ(test_mat_rand.get_elem(i, j).get_scalar(), h_mat_rand[i+j*m_rand]);
            }
        }

        free(h_mat_rand);

    }

    void TestBadDynamicMemConstruction() {

        double *h_mat = nullptr;

        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() { MatrixDense<double>(TestBase::bundle, h_mat, -1, 4);
            }
        );

        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() { MatrixDense<double>(TestBase::bundle, h_mat, 4, -2 ); }
        );

    }

    template <typename T>
    void TestDynamicMemCopyToPtr() {
    
        const int m_manual(2);
        const int n_manual(3);

        MatrixDense<T> mat_manual(
            TestBase::bundle,
            {{static_cast<T>(-5), static_cast<T>(100), static_cast<T>(-20)},
             {static_cast<T>(3), static_cast<T>(3.5), static_cast<T>(3)}}
        );

        T *h_mat_manual = static_cast<T *>(malloc(m_manual*n_manual*sizeof(T)));
        mat_manual.copy_data_to_ptr(h_mat_manual, m_manual, n_manual);

        ASSERT_EQ(h_mat_manual[0+0*m_manual], static_cast<T>(-5));
        ASSERT_EQ(h_mat_manual[1+0*m_manual], static_cast<T>(3));
        ASSERT_EQ(h_mat_manual[0+1*m_manual], static_cast<T>(100));
        ASSERT_EQ(h_mat_manual[1+1*m_manual], static_cast<T>(3.5));
        ASSERT_EQ(h_mat_manual[0+2*m_manual], static_cast<T>(-20));
        ASSERT_EQ(h_mat_manual[1+2*m_manual], static_cast<T>(3));

        free(h_mat_manual);
    
        const int m_rand(4);
        const int n_rand(5);

        MatrixDense<T> mat_rand(
            TestBase::bundle,
            {{static_cast<T>(rand()), static_cast<T>(rand()), static_cast<T>(rand()),
              static_cast<T>(rand()), static_cast<T>(rand())},
             {static_cast<T>(rand()), static_cast<T>(rand()), static_cast<T>(rand()),
              static_cast<T>(rand()), static_cast<T>(rand())},
             {static_cast<T>(rand()), static_cast<T>(rand()), static_cast<T>(rand()),
              static_cast<T>(rand()), static_cast<T>(rand())},
             {static_cast<T>(rand()), static_cast<T>(rand()), static_cast<T>(rand()),
              static_cast<T>(rand()), static_cast<T>(rand())}}
        );

        T *h_mat_rand = static_cast<T *>(malloc(m_rand*n_rand*sizeof(T)));
        mat_rand.copy_data_to_ptr(h_mat_rand, m_rand, n_rand);

        for (int i=0; i<m_rand; ++i) {
            for (int j=0; j<n_rand; ++j) {
                ASSERT_EQ(h_mat_rand[i+j*m_rand], mat_rand.get_elem(i, j).get_scalar());
            }
        }

        free(h_mat_rand);

    }

    void TestBadDynamicMemCopyToPtr() {

        const int m_rand(4);
        const int n_rand(5);
        MatrixDense<double> mat_rand(MatrixDense<double>::Random(TestBase::bundle, m_rand, n_rand));
        double *h_mat_rand = static_cast<double *>(malloc(m_rand*n_rand*sizeof(double)));
        
        auto try_row_too_small = [=]() { mat_rand.copy_data_to_ptr(h_mat_rand, m_rand-2, n_rand); };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_row_too_small);

        auto try_row_too_large = [=]() { mat_rand.copy_data_to_ptr(h_mat_rand, m_rand+2, n_rand); };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_row_too_large);

        auto try_col_too_small = [=]() { mat_rand.copy_data_to_ptr(h_mat_rand, m_rand, n_rand-2); };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_col_too_small);

        auto try_col_too_large = [=]() { mat_rand.copy_data_to_ptr(h_mat_rand, m_rand, n_rand+2); };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_col_too_large);

        auto try_match_wrong_dim_row = [=]() { mat_rand.copy_data_to_ptr(h_mat_rand, n_rand, n_rand); };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_match_wrong_dim_row);

        auto try_match_wrong_dim_col = [=]() { mat_rand.copy_data_to_ptr(h_mat_rand, m_rand, m_rand); };
        CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, try_match_wrong_dim_col);

        free(h_mat_rand);

    }

    template <typename T>
    void TestRandomMatrixCreation() {

        // Just test gives right size and numbers aren't generally the same
        // will fail with very low probability (check middle numbers are different
        // from 5 adjacent above and below)
        constexpr int m_rand(40);
        constexpr int n_rand(40);
        MatrixDense<T> test_rand(MatrixDense<T>::Random(TestBase::bundle, m_rand, n_rand));
        ASSERT_EQ(test_rand.rows(), m_rand);
        ASSERT_EQ(test_rand.cols(), n_rand);
        for (int i=1; i<m_rand-1; ++i) {
            for (int j=1; j<n_rand-1; ++j) {
                ASSERT_TRUE(
                    ((test_rand.get_elem(i, j).get_scalar() != test_rand.get_elem(i-1, j).get_scalar()) ||
                     (test_rand.get_elem(i, j).get_scalar() != test_rand.get_elem(i+1, j).get_scalar()) ||
                     (test_rand.get_elem(i, j).get_scalar() != test_rand.get_elem(i, j-1).get_scalar()) ||
                     (test_rand.get_elem(i, j).get_scalar() != test_rand.get_elem(i, j+1).get_scalar()))
                );
            }
        }

    }

    template <typename T>
    void TestBlock() {

        const MatrixDense<T> const_mat (
            TestBase::bundle,
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
            TestBase::bundle,
            {{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4)},
             {static_cast<T>(6), static_cast<T>(7), static_cast<T>(8), static_cast<T>(9)},
             {static_cast<T>(11), static_cast<T>(12), static_cast<T>(13), static_cast<T>(14)}}
        );
        ASSERT_MATRIX_EQ(mat_0_0_3_4, test_0_0_3_4);

        // Test MatrixDense cast/access for block 1, 2, 3, 1
        MatrixDense<T> mat_1_2_3_1(mat.get_block(1, 2, 3, 1));
        MatrixDense<T> test_1_2_3_1(
            TestBase::bundle,
            {{static_cast<T>(8)},
             {static_cast<T>(13)},
             {static_cast<T>(18)}}
        );
        ASSERT_MATRIX_EQ(mat_1_2_3_1, test_1_2_3_1);

        // Test MatrixDense cast/access for block 0, 0, 3, 4
        MatrixDense<T> mat_0_0_3_4_copy(mat.get_block(0, 0, 3, 4).copy_to_mat());
        ASSERT_MATRIX_EQ(mat_0_0_3_4_copy, test_0_0_3_4);

        // Test MatrixDense cast/access for block 1, 2, 3, 1
        MatrixDense<T> mat_1_2_3_1_copy(mat.get_block(1, 2, 3, 1).copy_to_mat());
        ASSERT_MATRIX_EQ(mat_1_2_3_1_copy, test_1_2_3_1);

        // Test assignment from MatrixDense
        mat = const_mat;
        MatrixDense<T> zero_2_3(MatrixDense<T>::Zero(TestBase::bundle, 2, 3));
        mat.get_block(1, 1, 2, 3).set_from_mat(zero_2_3);
        MatrixDense<T> test_assign_2_3(
            TestBase::bundle,
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
            TestBase::bundle,
            {static_cast<T>(1),
             static_cast<T>(1),
             static_cast<T>(1)}
        );
        mat.get_block(1, 4, 3, 1).set_from_vec(assign_vec);
        MatrixDense<T> test_assign_1_4(
            TestBase::bundle,
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

    void TestCast() {
        
        constexpr int m(10);
        constexpr int n(15);

        MatrixDense<double> mat_dbl(MatrixDense<double>::Random(TestBase::bundle, m, n));

        MatrixDense<double> dbl_to_dbl(mat_dbl.template cast<double>());
        ASSERT_MATRIX_EQ(dbl_to_dbl, mat_dbl);

        MatrixDense<float> dbl_to_sgl(mat_dbl.template cast<float>());
        ASSERT_EQ(dbl_to_sgl.rows(), m);
        ASSERT_EQ(dbl_to_sgl.cols(), n);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_NEAR(
                    dbl_to_sgl.get_elem(i, j).get_scalar(),
                    static_cast<float>(mat_dbl.get_elem(i, j).get_scalar()),
                    min_1_mag(static_cast<float>(mat_dbl.get_elem(i, j).get_scalar()))*
                        Tol<float>::roundoff_T()
                );
            }
        }

        MatrixDense<__half> dbl_to_hlf(mat_dbl.template cast<__half>());
        ASSERT_EQ(dbl_to_hlf.rows(), m);
        ASSERT_EQ(dbl_to_hlf.cols(), n);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_NEAR(
                    dbl_to_hlf.get_elem(i, j).get_scalar(),
                    static_cast<__half>(mat_dbl.get_elem(i, j).get_scalar()),
                    min_1_mag(static_cast<__half>(mat_dbl.get_elem(i, j).get_scalar()))*
                        Tol<__half>::roundoff_T()
                );
            }
        }

        MatrixDense<float> mat_sgl(MatrixDense<float>::Random(TestBase::bundle, m, n));

        MatrixDense<float> sgl_to_sgl(mat_sgl.template cast<float>());
        ASSERT_MATRIX_EQ(sgl_to_sgl, mat_sgl);
    
        MatrixDense<double> sgl_to_dbl(mat_sgl.template cast<double>());
        ASSERT_EQ(sgl_to_dbl.rows(), m);
        ASSERT_EQ(sgl_to_dbl.cols(), n);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_NEAR(
                    sgl_to_dbl.get_elem(i, j).get_scalar(),
                    static_cast<double>(mat_sgl.get_elem(i, j).get_scalar()),
                    min_1_mag(static_cast<double>(mat_sgl.get_elem(i, j).get_scalar()))*
                        static_cast<double>(Tol<float>::roundoff_T())
                );
            }
        }

        MatrixDense<__half> sgl_to_hlf(mat_sgl.template cast<__half>());
        ASSERT_EQ(sgl_to_hlf.rows(), m);
        ASSERT_EQ(sgl_to_hlf.cols(), n);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_NEAR(
                    sgl_to_hlf.get_elem(i, j).get_scalar(),
                    static_cast<__half>(mat_sgl.get_elem(i, j).get_scalar()),
                    min_1_mag(static_cast<__half>(mat_sgl.get_elem(i, j).get_scalar()))*
                        Tol<__half>::roundoff_T()
                );
            }
        }

        MatrixDense<__half> mat_hlf(MatrixDense<__half>::Random(TestBase::bundle, m, n));

        MatrixDense<__half> hlf_to_hlf(mat_hlf.template cast<__half>());
        ASSERT_MATRIX_EQ(hlf_to_hlf, mat_hlf);

        MatrixDense<float> hlf_to_sgl(mat_hlf.template cast<float>());
        ASSERT_EQ(hlf_to_sgl.rows(), m);
        ASSERT_EQ(hlf_to_sgl.cols(), n);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_NEAR(
                    hlf_to_sgl.get_elem(i, j).get_scalar(),
                    static_cast<float>(mat_hlf.get_elem(i, j).get_scalar()),
                    min_1_mag(static_cast<double>(mat_hlf.get_elem(i, j).get_scalar()))*
                        static_cast<float>(Tol<__half>::roundoff_T())
                );
            }
        }

        MatrixDense<double> hlf_to_dbl(mat_hlf.template cast<double>());
        ASSERT_EQ(hlf_to_dbl.rows(), m);
        ASSERT_EQ(hlf_to_dbl.cols(), n);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_NEAR(
                    hlf_to_dbl.get_elem(i, j).get_scalar(),
                    static_cast<double>(mat_hlf.get_elem(i, j).get_scalar()),
                    min_1_mag(static_cast<double>(mat_hlf.get_elem(i, j).get_scalar()))*
                        static_cast<double>(Tol<__half>::roundoff_T())
                );
            }
        }

    }

};

TEST_F(MatrixDense_Test, TestCoeffAccess) {
    TestCoeffAccess<__half>();
    TestCoeffAccess<float>();
    TestCoeffAccess<double>();
}

TEST_F(MatrixDense_Test, TestBadCoeffAccess) {
    TestBadCoeffAccess();
}

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

TEST_F(MatrixDense_Test, TestBadConstruction) {
    TestBadConstruction();
}

TEST_F(MatrixDense_Test, TestListInitialization) {
    TestListInitialization<__half>();
    TestListInitialization<float>();
    TestListInitialization<double>();
}

TEST_F(MatrixDense_Test, TestBadListInitialization) {
    TestBadListInitialization();
}

TEST_F(MatrixDense_Test, TestNonZeros) {
    TestNonZeros<__half>();
    TestNonZeros<float>();
    TestNonZeros<double>();
}

TEST_F(MatrixDense_Test, TestPrintAndInfoString) {
    TestPrintAndInfoString();
}

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

TEST_F(MatrixDense_Test, TestDynamicMemConstruction) {
    TestDynamicMemConstruction<__half>();
    TestDynamicMemConstruction<float>();
    TestDynamicMemConstruction<double>();
}

TEST_F(MatrixDense_Test, TestBadDynamicMemConstruction) {
    TestBadDynamicMemConstruction();
}

TEST_F(MatrixDense_Test, TestDynamicMemCopyToPtr) {
    TestDynamicMemCopyToPtr<__half>();
    TestDynamicMemCopyToPtr<float>();
    TestDynamicMemCopyToPtr<double>();
}

TEST_F(MatrixDense_Test, TestBadDynamicMemCopyToPtr) {
    TestBadDynamicMemCopyToPtr();
}

TEST_F(MatrixDense_Test, TestZeroMatrixCreation) {
    TestZeroMatrixCreation<__half>();
    TestZeroMatrixCreation<float>();
    TestZeroMatrixCreation<double>();
}

TEST_F(MatrixDense_Test, TestOnesMatrixCreation) {
    TestOnesMatrixCreation<__half>();
    TestOnesMatrixCreation<float>();
    TestOnesMatrixCreation<double>();
}

TEST_F(MatrixDense_Test, TestIdentityMatrixCreation) {
    TestIdentityMatrixCreation<__half>();
    TestIdentityMatrixCreation<float>();
    TestIdentityMatrixCreation<double>();
}

TEST_F(MatrixDense_Test, TestRandomMatrixCreation) {
    TestRandomMatrixCreation<__half>();
    TestRandomMatrixCreation<float>();
    TestRandomMatrixCreation<double>();
}

TEST_F(MatrixDense_Test, TestCol) {
    TestCol<__half>();
    TestCol<float>();
    TestCol<double>();
}

TEST_F(MatrixDense_Test, TestBadCol) {
    TestBadCol();
}

TEST_F(MatrixDense_Test, TestBlock) {
    TestBlock<__half>();
    TestBlock<float>();
    TestBlock<double>();
}

TEST_F(MatrixDense_Test, TestBadBlock) { 
    TestBadBlock();
}

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

TEST_F(MatrixDense_Test, TestCast) {
    TestCast();
}

TEST_F(MatrixDense_Test, TestBadCast) {
    TestBadCast();
}

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