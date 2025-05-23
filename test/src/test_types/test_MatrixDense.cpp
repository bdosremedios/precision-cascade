#include "test_Matrix.h"

#include "tools/abs.h"
#include "types/MatrixDense/MatrixDense.h"

class MatrixDense_Test: public Matrix_Test<MatrixDense>
{
public:

    template <typename TPrecision>
    void TestDynamicMemConstruction() {
    
        const int m_manual(2);
        const int n_manual(3);
        TPrecision *h_mat_manual = static_cast<TPrecision *>(
            malloc(m_manual*n_manual*sizeof(TPrecision))
        );
        h_mat_manual[0+0*m_manual] = static_cast<TPrecision>(-5);
        h_mat_manual[1+0*m_manual] = static_cast<TPrecision>(3);
        h_mat_manual[0+1*m_manual] = static_cast<TPrecision>(100);
        h_mat_manual[1+1*m_manual] = static_cast<TPrecision>(3.5);
        h_mat_manual[0+2*m_manual] = static_cast<TPrecision>(-20);
        h_mat_manual[1+2*m_manual] = static_cast<TPrecision>(3);

        MatrixDense<TPrecision> test_mat_manual(
            TestBase::bundle, h_mat_manual, m_manual, n_manual
        );

        MatrixDense<TPrecision> target_mat_manual(
            TestBase::bundle,
            {{static_cast<TPrecision>(-5), static_cast<TPrecision>(100),
              static_cast<TPrecision>(-20)},
             {static_cast<TPrecision>(3), static_cast<TPrecision>(3.5),
              static_cast<TPrecision>(3)}}
        );

        ASSERT_MATRIX_EQ(test_mat_manual, target_mat_manual);

        free(h_mat_manual);
    
        const int m_rand(4);
        const int n_rand(5);
        TPrecision *h_mat_rand = static_cast<TPrecision *>(
            malloc(m_rand*n_rand*sizeof(TPrecision))
        );
        for (int i=0; i<m_rand; ++i) {
            for (int j=0; j<n_rand; ++j) {
                h_mat_rand[i+j*m_rand] = static_cast<TPrecision>(rand());
            }
        }

        MatrixDense<TPrecision> test_mat_rand(
            TestBase::bundle, h_mat_rand, m_rand, n_rand
        );

        ASSERT_EQ(test_mat_rand.rows(), m_rand);
        ASSERT_EQ(test_mat_rand.cols(), n_rand);
        for (int i=0; i<m_rand; ++i) {
            for (int j=0; j<n_rand; ++j) {
                ASSERT_EQ(
                    test_mat_rand.get_elem(i, j).get_scalar(),
                    h_mat_rand[i+j*m_rand]
                );
            }
        }

        free(h_mat_rand);

    }

    void TestBadDynamicMemConstruction() {

        double *h_mat = nullptr;

        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() { MatrixDense<double>(TestBase::bundle, h_mat, -1, 4); }
        );

        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() { MatrixDense<double>(TestBase::bundle, h_mat, 4, -2 ); }
        );

    }

    template <typename TPrecision>
    void TestDynamicMemCopyToPtr() {
    
        const int m_manual(2);
        const int n_manual(3);

        MatrixDense<TPrecision> mat_manual(
            TestBase::bundle,
            {{static_cast<TPrecision>(-5), static_cast<TPrecision>(100),
              static_cast<TPrecision>(-20)},
             {static_cast<TPrecision>(3), static_cast<TPrecision>(3.5),
              static_cast<TPrecision>(3)}}
        );

        TPrecision *h_mat_manual = static_cast<TPrecision *>(
            malloc(m_manual*n_manual*sizeof(TPrecision))
        );
        mat_manual.copy_data_to_ptr(h_mat_manual, m_manual, n_manual);

        ASSERT_EQ(h_mat_manual[0+0*m_manual], static_cast<TPrecision>(-5));
        ASSERT_EQ(h_mat_manual[1+0*m_manual], static_cast<TPrecision>(3));
        ASSERT_EQ(h_mat_manual[0+1*m_manual], static_cast<TPrecision>(100));
        ASSERT_EQ(h_mat_manual[1+1*m_manual], static_cast<TPrecision>(3.5));
        ASSERT_EQ(h_mat_manual[0+2*m_manual], static_cast<TPrecision>(-20));
        ASSERT_EQ(h_mat_manual[1+2*m_manual], static_cast<TPrecision>(3));

        free(h_mat_manual);
    
        const int m_rand(4);
        const int n_rand(5);

        MatrixDense<TPrecision> mat_rand(
            TestBase::bundle,
            {{static_cast<TPrecision>(rand()), static_cast<TPrecision>(rand()),
              static_cast<TPrecision>(rand()), static_cast<TPrecision>(rand()),
              static_cast<TPrecision>(rand())},
             {static_cast<TPrecision>(rand()), static_cast<TPrecision>(rand()),
              static_cast<TPrecision>(rand()), static_cast<TPrecision>(rand()),
              static_cast<TPrecision>(rand())},
             {static_cast<TPrecision>(rand()), static_cast<TPrecision>(rand()),
              static_cast<TPrecision>(rand()), static_cast<TPrecision>(rand()),
              static_cast<TPrecision>(rand())},
             {static_cast<TPrecision>(rand()), static_cast<TPrecision>(rand()),
              static_cast<TPrecision>(rand()), static_cast<TPrecision>(rand()),
              static_cast<TPrecision>(rand())}}
        );

        TPrecision *h_mat_rand = static_cast<TPrecision *>(
            malloc(m_rand*n_rand*sizeof(TPrecision))
        );
        mat_rand.copy_data_to_ptr(h_mat_rand, m_rand, n_rand);

        for (int i=0; i<m_rand; ++i) {
            for (int j=0; j<n_rand; ++j) {
                ASSERT_EQ(
                    h_mat_rand[i+j*m_rand],
                    mat_rand.get_elem(i, j).get_scalar()
                );
            }
        }

        free(h_mat_rand);

    }

    void TestBadDynamicMemCopyToPtr() {

        const int m_rand(4);
        const int n_rand(5);
        MatrixDense<double> mat_rand(MatrixDense<double>::Random(
            TestBase::bundle, m_rand, n_rand
        ));
        double *h_mat_rand = static_cast<double *>(
            malloc(m_rand*n_rand*sizeof(double))
        );
        
        auto try_row_too_small = [=]() {
            mat_rand.copy_data_to_ptr(h_mat_rand, m_rand-2, n_rand);
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors, try_row_too_small);

        auto try_row_too_large = [=]() {
            mat_rand.copy_data_to_ptr(h_mat_rand, m_rand+2, n_rand);
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors, try_row_too_large);

        auto try_col_too_small = [=]() {
            mat_rand.copy_data_to_ptr(h_mat_rand, m_rand, n_rand-2);
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors, try_col_too_small);

        auto try_col_too_large = [=]() {
            mat_rand.copy_data_to_ptr(h_mat_rand, m_rand, n_rand+2);
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors, try_col_too_large);

        auto try_match_wrong_dim_row = [=]() {
            mat_rand.copy_data_to_ptr(h_mat_rand, n_rand, n_rand);
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors, try_match_wrong_dim_row);

        auto try_match_wrong_dim_col = [=]() {
            mat_rand.copy_data_to_ptr(h_mat_rand, m_rand, m_rand);
        };
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors, try_match_wrong_dim_col);

        free(h_mat_rand);

    }

    template <typename TPrecision>
    void TestRandomMatrixCreation() {

        // Just test gives right size and numbers aren't generally the same
        // will fail with very low probability (check middle numbers are
        // different from 5 adjacent above and below)
        constexpr int m_rand(40);
        constexpr int n_rand(40);
        MatrixDense<TPrecision> test_rand(MatrixDense<TPrecision>::Random(
            TestBase::bundle, m_rand, n_rand
        ));
        ASSERT_EQ(test_rand.rows(), m_rand);
        ASSERT_EQ(test_rand.cols(), n_rand);
        for (int i=1; i<m_rand-1; ++i) {
            for (int j=1; j<n_rand-1; ++j) {
                ASSERT_TRUE(
                    ((test_rand.get_elem(i, j).get_scalar() !=
                      test_rand.get_elem(i-1, j).get_scalar()) ||
                     (test_rand.get_elem(i, j).get_scalar() !=
                      test_rand.get_elem(i+1, j).get_scalar()) ||
                     (test_rand.get_elem(i, j).get_scalar() !=
                      test_rand.get_elem(i, j-1).get_scalar()) ||
                     (test_rand.get_elem(i, j).get_scalar() !=
                      test_rand.get_elem(i, j+1).get_scalar()))
                );
            }
        }

    }

    template <typename TPrecision>
    void TestRandomUTMatrixCreation() {

        // Test gives right size and numbers aren't generally the same
        // will fail with very low probability (check middle numbers are
        // different from 5 adjacent above and below)
        constexpr int m_rand(40);
        constexpr int n_rand(40);
        MatrixDense<TPrecision> test_rand(MatrixDense<TPrecision>::Random_UT(
            TestBase::bundle, m_rand, n_rand
        ));
        ASSERT_EQ(test_rand.rows(), m_rand);
        ASSERT_EQ(test_rand.cols(), n_rand);
        for (int i=1; i<m_rand-1; ++i) {
            for (int j=i+1; j<n_rand-1; ++j) {
                ASSERT_TRUE(
                    ((test_rand.get_elem(i, j).get_scalar() !=
                      test_rand.get_elem(i-1, j).get_scalar()) ||
                     (test_rand.get_elem(i, j).get_scalar() !=
                      test_rand.get_elem(i+1, j).get_scalar()) ||
                     (test_rand.get_elem(i, j).get_scalar() !=
                      test_rand.get_elem(i, j-1).get_scalar()) ||
                     (test_rand.get_elem(i, j).get_scalar() !=
                      test_rand.get_elem(i, j+1).get_scalar()))
                );
            }
        }

        // Check non-zero diagonal
        for (int i=0; i<m_rand; ++i) {
            ASSERT_FALSE(
                test_rand.get_elem(i, i).get_scalar() ==
                static_cast<TPrecision>(0.)
            );
        }

        // Check zero below diagonal
        for (int i=0; i<m_rand; ++i) {
            for (int j=0; j<i; ++j) {
                ASSERT_EQ(
                    test_rand.get_elem(i, j).get_scalar(),
                    static_cast<TPrecision>(0.)
                );
            }
        }

    }

    template <typename TPrecision>
    void TestRandomLTMatrixCreation() {

        // Test gives right size and numbers aren't generally the same
        // will fail with very low probability (check middle numbers are
        // different from 5 adjacent above and below)
        constexpr int m_rand(40);
        constexpr int n_rand(40);
        MatrixDense<TPrecision> test_rand(MatrixDense<TPrecision>::Random_LT(
            TestBase::bundle, m_rand, n_rand
        ));
        ASSERT_EQ(test_rand.rows(), m_rand);
        ASSERT_EQ(test_rand.cols(), n_rand);
        for (int i=1; i<m_rand-1; ++i) {
            for (int j=1; j<i; ++j) {
                ASSERT_TRUE(
                    ((test_rand.get_elem(i, j).get_scalar() !=
                      test_rand.get_elem(i-1, j).get_scalar()) ||
                     (test_rand.get_elem(i, j).get_scalar() !=
                      test_rand.get_elem(i+1, j).get_scalar()) ||
                     (test_rand.get_elem(i, j).get_scalar() !=
                      test_rand.get_elem(i, j-1).get_scalar()) ||
                     (test_rand.get_elem(i, j).get_scalar() !=
                      test_rand.get_elem(i, j+1).get_scalar()))
                );
            }
        }

        // Check non-zero diagonal
        for (int i=0; i<m_rand; ++i) {
            ASSERT_FALSE(
                test_rand.get_elem(i, i).get_scalar() ==
                static_cast<TPrecision>(0.)
            );
        }

        // Check zero above diagonal
        for (int i=0; i<m_rand; ++i) {
            for (int j=i+1; j<n_rand; ++j) {
                ASSERT_EQ(
                    test_rand.get_elem(i, j).get_scalar(),
                    static_cast<TPrecision>(0.)
                );
            }
        }

    }

    template <typename TPrecision>
    void TestBlock() {

        const MatrixDense<TPrecision> const_mat (
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(2),
              static_cast<TPrecision>(3), static_cast<TPrecision>(4),
              static_cast<TPrecision>(5)},
             {static_cast<TPrecision>(6), static_cast<TPrecision>(7),
              static_cast<TPrecision>(8), static_cast<TPrecision>(9),
              static_cast<TPrecision>(10)},
             {static_cast<TPrecision>(11), static_cast<TPrecision>(12),
              static_cast<TPrecision>(13), static_cast<TPrecision>(14),
              static_cast<TPrecision>(15)},
             {static_cast<TPrecision>(16), static_cast<TPrecision>(17),
              static_cast<TPrecision>(18), static_cast<TPrecision>(19),
              static_cast<TPrecision>(20)}}
        );
        MatrixDense<TPrecision> mat(const_mat);

        // Test copy constructor and access for block 0, 0, 4, 2
        typename MatrixDense<TPrecision>::Block blk_0_0_4_2(
            mat.get_block(0, 0, 4, 2)
        );
        ASSERT_EQ(
            blk_0_0_4_2.get_elem(0, 0).get_scalar(),
            static_cast<TPrecision>(1)
        );
        ASSERT_EQ(
            blk_0_0_4_2.get_elem(1, 0).get_scalar(),
            static_cast<TPrecision>(6)
        );
        ASSERT_EQ(
            blk_0_0_4_2.get_elem(2, 0).get_scalar(),
            static_cast<TPrecision>(11)
        );
        ASSERT_EQ(
            blk_0_0_4_2.get_elem(3, 0).get_scalar(),
            static_cast<TPrecision>(16)
        );
        ASSERT_EQ(
            blk_0_0_4_2.get_elem(0, 1).get_scalar(),
            static_cast<TPrecision>(2)
        );
        ASSERT_EQ(
            blk_0_0_4_2.get_elem(1, 1).get_scalar(),
            static_cast<TPrecision>(7)
        );
        ASSERT_EQ(
            blk_0_0_4_2.get_elem(2, 1).get_scalar(),
            static_cast<TPrecision>(12)
        );
        ASSERT_EQ(
            blk_0_0_4_2.get_elem(3, 1).get_scalar(),
            static_cast<TPrecision>(17)
        );

        // Test copy constructor and access for block 2, 1, 2, 3
        typename MatrixDense<TPrecision>::Block blk_2_1_2_3(
            mat.get_block(2, 1, 2, 3)
        );
        ASSERT_EQ(
            blk_2_1_2_3.get_elem(0, 0).get_scalar(),
            static_cast<TPrecision>(12)
        );
        ASSERT_EQ(
            blk_2_1_2_3.get_elem(0, 1).get_scalar(),
            static_cast<TPrecision>(13)
        );
        ASSERT_EQ(
            blk_2_1_2_3.get_elem(0, 2).get_scalar(),
            static_cast<TPrecision>(14)
        );
        ASSERT_EQ(
            blk_2_1_2_3.get_elem(1, 0).get_scalar(),
            static_cast<TPrecision>(17)
        );
        ASSERT_EQ(
            blk_2_1_2_3.get_elem(1, 1).get_scalar(),
            static_cast<TPrecision>(18)
        );
        ASSERT_EQ(
            blk_2_1_2_3.get_elem(1, 2).get_scalar(),
            static_cast<TPrecision>(19)
        );

        // Test MatrixDense cast/access for block 0, 0, 3, 4
        MatrixDense<TPrecision> mat_0_0_3_4(
            mat.get_block(0, 0, 3, 4).copy_to_mat()
        );
        MatrixDense<TPrecision> test_0_0_3_4(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(2),
              static_cast<TPrecision>(3), static_cast<TPrecision>(4)},
             {static_cast<TPrecision>(6), static_cast<TPrecision>(7),
              static_cast<TPrecision>(8), static_cast<TPrecision>(9)},
             {static_cast<TPrecision>(11), static_cast<TPrecision>(12),
              static_cast<TPrecision>(13), static_cast<TPrecision>(14)}}
        );
        ASSERT_MATRIX_EQ(mat_0_0_3_4, test_0_0_3_4);

        // Test MatrixDense cast/access for block 1, 2, 3, 1
        MatrixDense<TPrecision> mat_1_2_3_1(
            mat.get_block(1, 2, 3, 1).copy_to_mat()
        );
        MatrixDense<TPrecision> test_1_2_3_1(
            TestBase::bundle,
            {{static_cast<TPrecision>(8)},
             {static_cast<TPrecision>(13)},
             {static_cast<TPrecision>(18)}}
        );
        ASSERT_MATRIX_EQ(mat_1_2_3_1, test_1_2_3_1);

        // Test MatrixDense cast/access for block 0, 0, 3, 4
        MatrixDense<TPrecision> mat_0_0_3_4_copy(
            mat.get_block(0, 0, 3, 4).copy_to_mat()
        );
        ASSERT_MATRIX_EQ(mat_0_0_3_4_copy, test_0_0_3_4);

        // Test MatrixDense cast/access for block 1, 2, 3, 1
        MatrixDense<TPrecision> mat_1_2_3_1_copy(
            mat.get_block(1, 2, 3, 1).copy_to_mat()
        );
        ASSERT_MATRIX_EQ(mat_1_2_3_1_copy, test_1_2_3_1);

        // Test assignment from MatrixDense
        mat = const_mat;
        MatrixDense<TPrecision> zero_2_3(MatrixDense<TPrecision>::Zero(
            TestBase::bundle, 2, 3
        ));
        mat.get_block(1, 1, 2, 3).set_from_mat(zero_2_3);
        MatrixDense<TPrecision> test_assign_2_3(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(2),
              static_cast<TPrecision>(3), static_cast<TPrecision>(4),
              static_cast<TPrecision>(5)},
             {static_cast<TPrecision>(6), static_cast<TPrecision>(0),
              static_cast<TPrecision>(0), static_cast<TPrecision>(0),
              static_cast<TPrecision>(10)},
             {static_cast<TPrecision>(11), static_cast<TPrecision>(0),
              static_cast<TPrecision>(0), static_cast<TPrecision>(0),
              static_cast<TPrecision>(15)},
             {static_cast<TPrecision>(16), static_cast<TPrecision>(17),
              static_cast<TPrecision>(18), static_cast<TPrecision>(19),
              static_cast<TPrecision>(20)}}
        );
        ASSERT_MATRIX_EQ(mat, test_assign_2_3);

        // Test assignment from Vector
        mat = const_mat;
        Vector<TPrecision> assign_vec(
            TestBase::bundle,
            {static_cast<TPrecision>(1),
             static_cast<TPrecision>(1),
             static_cast<TPrecision>(1)}
        );
        mat.get_block(1, 4, 3, 1).set_from_vec(assign_vec);
        MatrixDense<TPrecision> test_assign_1_4(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(2),
              static_cast<TPrecision>(3), static_cast<TPrecision>(4),
              static_cast<TPrecision>(5)},
             {static_cast<TPrecision>(6), static_cast<TPrecision>(7),
              static_cast<TPrecision>(8), static_cast<TPrecision>(9),
              static_cast<TPrecision>(1)},
             {static_cast<TPrecision>(11), static_cast<TPrecision>(12),
              static_cast<TPrecision>(13), static_cast<TPrecision>(14),
              static_cast<TPrecision>(1)},
             {static_cast<TPrecision>(16), static_cast<TPrecision>(17),
              static_cast<TPrecision>(18), static_cast<TPrecision>(19),
              static_cast<TPrecision>(1)}}
        );
        ASSERT_MATRIX_EQ(mat, test_assign_1_4);

    }

    template <typename TPrecision>
    void TestRandomMatVec() {

        // Test random
        const int m_rand(3);
        const int n_rand(4);
        MatrixDense<TPrecision> rand_mat(MatrixDense<TPrecision>::Random(
            TestBase::bundle, m_rand, n_rand
        ));
        ASSERT_VECTOR_NEAR(
            rand_mat*Vector<TPrecision>(
                TestBase::bundle,
                {static_cast<TPrecision>(1), static_cast<TPrecision>(0),
                 static_cast<TPrecision>(0), static_cast<TPrecision>(0)}
            ),
            Vector<TPrecision>(
                TestBase::bundle,
                {rand_mat.get_elem(0, 0).get_scalar(),
                 rand_mat.get_elem(1, 0).get_scalar(),
                 rand_mat.get_elem(2, 0).get_scalar()}
            ),
            Vector<TPrecision>(
                TestBase::bundle,
                {abs_ns::abs(rand_mat.get_elem(0, 0).get_scalar()),
                 abs_ns::abs(rand_mat.get_elem(1, 0).get_scalar()),
                 abs_ns::abs(rand_mat.get_elem(2, 0).get_scalar())}
            )*Tol<TPrecision>::gamma_T(4)
        );
        ASSERT_VECTOR_NEAR(
            rand_mat*Vector<TPrecision>(
                TestBase::bundle,
                {static_cast<TPrecision>(0), static_cast<TPrecision>(1),
                 static_cast<TPrecision>(0), static_cast<TPrecision>(0)}
            ),
            Vector<TPrecision>(
                TestBase::bundle,
                {rand_mat.get_elem(0, 1).get_scalar(),
                 rand_mat.get_elem(1, 1).get_scalar(),
                 rand_mat.get_elem(2, 1).get_scalar()}
            ),
            Vector<TPrecision>(
                TestBase::bundle,
                {abs_ns::abs(rand_mat.get_elem(0, 1).get_scalar()),
                 abs_ns::abs(rand_mat.get_elem(1, 1).get_scalar()),
                 abs_ns::abs(rand_mat.get_elem(2, 1).get_scalar())}
            )*Tol<TPrecision>::gamma_T(4)
        );
        ASSERT_VECTOR_NEAR(
            rand_mat*Vector<TPrecision>(
                TestBase::bundle,
                {static_cast<TPrecision>(0), static_cast<TPrecision>(0),
                 static_cast<TPrecision>(1), static_cast<TPrecision>(0)}
            ),
            Vector<TPrecision>(
                TestBase::bundle,
                {rand_mat.get_elem(0, 2).get_scalar(),
                 rand_mat.get_elem(1, 2).get_scalar(),
                 rand_mat.get_elem(2, 2).get_scalar()}
            ),
            Vector<TPrecision>(
                TestBase::bundle,
                {abs_ns::abs(rand_mat.get_elem(0, 2).get_scalar()),
                 abs_ns::abs(rand_mat.get_elem(1, 2).get_scalar()),
                 abs_ns::abs(rand_mat.get_elem(2, 2).get_scalar())}
            )*Tol<TPrecision>::gamma_T(4)
        );
        ASSERT_VECTOR_NEAR(
            rand_mat*Vector<TPrecision>(
                TestBase::bundle,
                {static_cast<TPrecision>(0), static_cast<TPrecision>(0),
                 static_cast<TPrecision>(0), static_cast<TPrecision>(1)}
            ),
            Vector<TPrecision>(
                TestBase::bundle,
                {rand_mat.get_elem(0, 3).get_scalar(),
                 rand_mat.get_elem(1, 3).get_scalar(),
                 rand_mat.get_elem(2, 3).get_scalar()}
            ),
            Vector<TPrecision>(
                TestBase::bundle,
                {abs_ns::abs(rand_mat.get_elem(0, 3).get_scalar()),
                 abs_ns::abs(rand_mat.get_elem(1, 3).get_scalar()),
                 abs_ns::abs(rand_mat.get_elem(2, 3).get_scalar())}
            )*Tol<TPrecision>::gamma_T(4)
        );
        ASSERT_VECTOR_NEAR(
            rand_mat*Vector<TPrecision>(
                TestBase::bundle,
                {static_cast<TPrecision>(-1),
                 static_cast<TPrecision>(-0.1),
                 static_cast<TPrecision>(0.01),
                 static_cast<TPrecision>(0.001)}
            ),
            Vector<TPrecision>(
                TestBase::bundle,
                {(static_cast<TPrecision>(-1) *
                  rand_mat.get_elem(0, 0).get_scalar() +
                  static_cast<TPrecision>(-0.1) *
                  rand_mat.get_elem(0, 1).get_scalar() +
                  static_cast<TPrecision>(0.01) *
                  rand_mat.get_elem(0, 2).get_scalar() +
                  static_cast<TPrecision>(0.001) *
                  rand_mat.get_elem(0, 3).get_scalar()),
                 (static_cast<TPrecision>(-1) *
                  rand_mat.get_elem(1, 0).get_scalar() +
                  static_cast<TPrecision>(-0.1) *
                  rand_mat.get_elem(1, 1).get_scalar() +
                  static_cast<TPrecision>(0.01) *
                  rand_mat.get_elem(1, 2).get_scalar()+
                  static_cast<TPrecision>(0.001) *
                  rand_mat.get_elem(1, 3).get_scalar()),
                 (static_cast<TPrecision>(-1) *
                  rand_mat.get_elem(2, 0).get_scalar() +
                  static_cast<TPrecision>(-0.1) *
                  rand_mat.get_elem(2, 1).get_scalar() +
                  static_cast<TPrecision>(0.01) *
                  rand_mat.get_elem(2, 2).get_scalar()+
                  static_cast<TPrecision>(0.001) *
                  rand_mat.get_elem(2, 3).get_scalar())}
            ),
            Vector<TPrecision>(
                TestBase::bundle,
                {(static_cast<TPrecision>(1) *
                  abs_ns::abs(rand_mat.get_elem(0, 0).get_scalar()) +
                  static_cast<TPrecision>(0.1) *
                  abs_ns::abs(rand_mat.get_elem(0, 1).get_scalar()) +
                  static_cast<TPrecision>(0.01) *
                  abs_ns::abs(rand_mat.get_elem(0, 2).get_scalar()) +
                  static_cast<TPrecision>(0.001) *
                  abs_ns::abs(rand_mat.get_elem(0, 3).get_scalar())),
                 (static_cast<TPrecision>(1) *
                  abs_ns::abs(rand_mat.get_elem(1, 0).get_scalar()) +
                  static_cast<TPrecision>(0.1) *
                  abs_ns::abs(rand_mat.get_elem(1, 1).get_scalar()) +
                  static_cast<TPrecision>(0.01) *
                  abs_ns::abs(rand_mat.get_elem(1, 2).get_scalar())+
                  static_cast<TPrecision>(0.001) *
                  abs_ns::abs(rand_mat.get_elem(1, 3).get_scalar())),
                 (static_cast<TPrecision>(1) *
                  abs_ns::abs(rand_mat.get_elem(2, 0).get_scalar()) +
                  static_cast<TPrecision>(0.1) *
                  abs_ns::abs(rand_mat.get_elem(2, 1).get_scalar()) +
                  static_cast<TPrecision>(0.01) *
                  abs_ns::abs(rand_mat.get_elem(2, 2).get_scalar())+
                  static_cast<TPrecision>(0.001) *
                  abs_ns::abs(rand_mat.get_elem(2, 3).get_scalar()))}
            )*Tol<TPrecision>::gamma_T(4)
        );

    }

    template <typename TPrecision>
    void TestSubsetcolsMatVec() {

        // Test manually
        MatrixDense<TPrecision> mat(
            TestBase::bundle,
            {{static_cast<TPrecision>(-1), static_cast<TPrecision>(2), static_cast<TPrecision>(-3)},
             {static_cast<TPrecision>(4), static_cast<TPrecision>(5), static_cast<TPrecision>(-6)},
             {static_cast<TPrecision>(7), static_cast<TPrecision>(-8), static_cast<TPrecision>(-9)},
             {static_cast<TPrecision>(-3), static_cast<TPrecision>(-2), static_cast<TPrecision>(-1)}}
        );
        Vector<TPrecision> mat_0(mat.get_col(0).copy_to_vec());
        Vector<TPrecision> mat_1(mat.get_col(1).copy_to_vec());
        Vector<TPrecision> mat_2(mat.get_col(2).copy_to_vec());

        Vector<TPrecision> vec_2_0_1(
            TestBase::bundle,
            {static_cast<TPrecision>(0), static_cast<TPrecision>(1)}
        );
        Vector<TPrecision> vec_2_1_0(
            TestBase::bundle,
            {static_cast<TPrecision>(1), static_cast<TPrecision>(0)}
        );
        Vector<TPrecision> vec_2_1_01(
            TestBase::bundle,
            {static_cast<TPrecision>(1), static_cast<TPrecision>(0.1)}
        );

        // Test multiplication of first 2 columns
        ASSERT_VECTOR_NEAR(
            mat.mult_subset_cols(0, 2, vec_2_0_1),
            mat_1,
            mat_1.abs()*Tol<TPrecision>::gamma_T(2)
        );
        ASSERT_VECTOR_NEAR(
            mat.mult_subset_cols(0, 2, vec_2_1_0),
            mat_0,
            mat_0.abs()*Tol<TPrecision>::gamma_T(2)
        );
        ASSERT_VECTOR_NEAR(
            mat.mult_subset_cols(0, 2, vec_2_1_01),
            mat_0+mat_1*Scalar<TPrecision>(static_cast<TPrecision>(0.1)),
            (
                mat_0.abs() +
                mat_1.abs()*Scalar<TPrecision>(static_cast<TPrecision>(0.1))
            )*Tol<TPrecision>::gamma_T(2)
        );

        // Test multiplication of last 2 columns
        ASSERT_VECTOR_NEAR(
            mat.mult_subset_cols(1, 2, vec_2_0_1),
            mat_2,
            mat_2.abs()*Tol<TPrecision>::gamma_T(2)
        );
        ASSERT_VECTOR_NEAR(
            mat.mult_subset_cols(1, 2, vec_2_1_0),
            mat_1,
            mat_1.abs()*Tol<TPrecision>::gamma_T(2)
        );
        ASSERT_VECTOR_NEAR(
            mat.mult_subset_cols(1, 2, vec_2_1_01),
            mat_1+mat_2*Scalar<TPrecision>(static_cast<TPrecision>(0.1)),
            (
                mat_1.abs() +
                mat_2.abs()*Scalar<TPrecision>(static_cast<TPrecision>(0.1))
            )*Tol<TPrecision>::gamma_T(2)
        );

        // Test multiplication of all columns
        Vector<TPrecision> vec_3_001_01_1(
            TestBase::bundle,
            {static_cast<TPrecision>(0.01), static_cast<TPrecision>(0.1),
             static_cast<TPrecision>(1.)}
        );
        ASSERT_VECTOR_NEAR(
            mat.mult_subset_cols(0, 3, vec_3_001_01_1),
            mat_0*Scalar<TPrecision>(static_cast<TPrecision>(0.01)) +
            mat_1*Scalar<TPrecision>(static_cast<TPrecision>(0.1)) +
            mat_2*Scalar<TPrecision>(static_cast<TPrecision>(1.)),
            (
                mat_0.abs()*Scalar<TPrecision>(static_cast<TPrecision>(0.01)) +
                mat_1.abs()*Scalar<TPrecision>(static_cast<TPrecision>(0.1)) +
                mat_2.abs()*Scalar<TPrecision>(static_cast<TPrecision>(1.))
            )*Tol<TPrecision>::gamma_T(2)
        );

        // Test multiplication of individual
        Vector<TPrecision> vec_1_2(
            TestBase::bundle, {static_cast<TPrecision>(2.)}
        );
        ASSERT_VECTOR_NEAR(
            mat.mult_subset_cols(0, 1, vec_1_2),
            mat_0*Scalar<TPrecision>(static_cast<TPrecision>(2.)),
            (
                mat_0.abs() *
                static_cast<TPrecision>(2.) *
                Tol<TPrecision>::roundoff_T()
            )
        );
        ASSERT_VECTOR_NEAR(
            mat.mult_subset_cols(1, 1, vec_1_2),
            mat_1*Scalar<TPrecision>(static_cast<TPrecision>(2.)),
            (
                mat_1.abs() *
                static_cast<TPrecision>(2.) *
                Tol<TPrecision>::roundoff_T()
            )
        );
        ASSERT_VECTOR_NEAR(
            mat.mult_subset_cols(2, 1, vec_1_2),
            mat_2*Scalar<TPrecision>(static_cast<TPrecision>(2.)),
            (
                mat_2.abs() *
                static_cast<TPrecision>(2.) *
                Tol<TPrecision>::roundoff_T()
            )
        );

    }

    template <typename TPrecision>
    void TestRandomSubsetcolsMatVec() {

        // Test manually
        MatrixDense<TPrecision> mat(MatrixDense<TPrecision>::Random(
            TestBase::bundle, 4, 3
        ));
        Vector<TPrecision> mat_0(mat.get_col(0).copy_to_vec());
        Vector<TPrecision> mat_1(mat.get_col(1).copy_to_vec());
        Vector<TPrecision> mat_2(mat.get_col(2).copy_to_vec());

        Vector<TPrecision> vec_2_0_1(
            TestBase::bundle,
            {static_cast<TPrecision>(0), static_cast<TPrecision>(1)}
        );
        Vector<TPrecision> vec_2_1_0(
            TestBase::bundle,
            {static_cast<TPrecision>(1), static_cast<TPrecision>(0)}
        );
        Vector<TPrecision> vec_2_1_01(
            TestBase::bundle,
            {static_cast<TPrecision>(1), static_cast<TPrecision>(0.1)}
        );

        // Test multiplication of first 2 columns
        ASSERT_VECTOR_NEAR(
            mat.mult_subset_cols(0, 2, vec_2_0_1),
            mat_1,
            mat_1.abs()*Tol<TPrecision>::gamma_T(2)
        );
        ASSERT_VECTOR_NEAR(
            mat.mult_subset_cols(0, 2, vec_2_1_0),
            mat_0,
            mat_0.abs()*Tol<TPrecision>::gamma_T(2)
        );
        ASSERT_VECTOR_NEAR(
            mat.mult_subset_cols(0, 2, vec_2_1_01),
            mat_0+mat_1*Scalar<TPrecision>(static_cast<TPrecision>(0.1)),
            (
                mat_0.abs() +
                mat_1.abs()*Scalar<TPrecision>(static_cast<TPrecision>(0.1))
            )*Tol<TPrecision>::gamma_T(2)
        );

        // Test multiplication of last 2 columns
        ASSERT_VECTOR_NEAR(
            mat.mult_subset_cols(1, 2, vec_2_0_1),
            mat_2,
            mat_2.abs()*Tol<TPrecision>::gamma_T(2)
        );
        ASSERT_VECTOR_NEAR(
            mat.mult_subset_cols(1, 2, vec_2_1_0),
            mat_1,
            mat_1.abs()*Tol<TPrecision>::gamma_T(2)
        );
        ASSERT_VECTOR_NEAR(
            mat.mult_subset_cols(1, 2, vec_2_1_01),
            mat_1+mat_2*Scalar<TPrecision>(static_cast<TPrecision>(0.1)),
            (
                mat_1.abs() +
                mat_2.abs()*Scalar<TPrecision>(static_cast<TPrecision>(0.1))
            )*Tol<TPrecision>::gamma_T(2)
        );

        // Test multiplication of all columns
        Vector<TPrecision> vec_3_001_01_1(
            TestBase::bundle,
            {static_cast<TPrecision>(0.01), static_cast<TPrecision>(0.1),
             static_cast<TPrecision>(1.)}
        );
        ASSERT_VECTOR_NEAR(
            mat.mult_subset_cols(0, 3, vec_3_001_01_1),
            mat_0*Scalar<TPrecision>(static_cast<TPrecision>(0.01)) +
            mat_1*Scalar<TPrecision>(static_cast<TPrecision>(0.1)) +
            mat_2*Scalar<TPrecision>(static_cast<TPrecision>(1.)),
            (
                mat_0.abs()*Scalar<TPrecision>(static_cast<TPrecision>(0.01)) +
                mat_1.abs()*Scalar<TPrecision>(static_cast<TPrecision>(0.1)) +
                mat_2.abs()*Scalar<TPrecision>(static_cast<TPrecision>(1.))
            )*Tol<TPrecision>::gamma_T(3)
        );

        // Test multiplication of individual
        Vector<TPrecision> vec_1_2(
            TestBase::bundle, {static_cast<TPrecision>(2.)}
        );
        ASSERT_VECTOR_NEAR(
            mat.mult_subset_cols(0, 1, vec_1_2),
            mat_0*Scalar<TPrecision>(static_cast<TPrecision>(2.)),
            mat_0.abs()*Tol<TPrecision>::roundoff_T()
        );
        ASSERT_VECTOR_NEAR(
            mat.mult_subset_cols(1, 1, vec_1_2),
            mat_1*Scalar<TPrecision>(static_cast<TPrecision>(2.)),
            mat_1.abs()*Tol<TPrecision>::roundoff_T()
        );
        ASSERT_VECTOR_NEAR(
            mat.mult_subset_cols(2, 1, vec_1_2),
            mat_2*Scalar<TPrecision>(static_cast<TPrecision>(2.)),
            mat_2.abs()*Tol<TPrecision>::roundoff_T()
        );

    }

    template <typename TPrecision>
    void TestLimitRandomSubsetcolsMatVec() {

        const int m(12);
        const int n(7);

        MatrixDense<TPrecision> mat(MatrixDense<TPrecision>::Random(
            TestBase::bundle, m, n
        ));
        Vector<TPrecision> vec(Vector<TPrecision>::Random(
            TestBase::bundle, n
        ));

        ASSERT_VECTOR_NEAR(
            mat.mult_subset_cols(0, n, vec),
            mat*vec,
            mat.abs()*vec.abs()*Tol<TPrecision>::gamma_T(n)
        );

    }

    template <typename TPrecision>
    void TestBadSubsetcolsMatVec() {

        MatrixDense<TPrecision> mat(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(2),
              static_cast<TPrecision>(3), static_cast<TPrecision>(4)},
             {static_cast<TPrecision>(5), static_cast<TPrecision>(6),
              static_cast<TPrecision>(7), static_cast<TPrecision>(8)},
             {static_cast<TPrecision>(9), static_cast<TPrecision>(10),
              static_cast<TPrecision>(11), static_cast<TPrecision>(12)}}
        );

        Vector<TPrecision> valid_vec(
            TestBase::bundle,
            {static_cast<TPrecision>(1), static_cast<TPrecision>(1),
             static_cast<TPrecision>(1)}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors, [=]() { mat.mult_subset_cols(-1, 3, valid_vec); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors, [=]() { mat.mult_subset_cols(4, 3, valid_vec); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors, [=]() { mat.mult_subset_cols(2, 3, valid_vec); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors, [=]() { mat.mult_subset_cols(1, -1, valid_vec); }
        );

        Vector<TPrecision> empty_vec(TestBase::bundle, {});
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,[=]() { mat.mult_subset_cols(0, 0, empty_vec); }
        );

        Vector<TPrecision> vec_too_small(
            TestBase::bundle,
            {static_cast<TPrecision>(1), static_cast<TPrecision>(1)}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() { mat.mult_subset_cols(0, 3, vec_too_small); }
        );

        Vector<TPrecision> vec_too_large(
            TestBase::bundle, 
            {static_cast<TPrecision>(1), static_cast<TPrecision>(1),
             static_cast<TPrecision>(1), static_cast<TPrecision>(1)}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() { mat.mult_subset_cols(0, 3, vec_too_large); }
        );
    
    }

    template <typename TPrecision>
    void TestRandomTransposeMatVec() {

        // Test random
        const int m_rand(3);
        const int n_rand(2);
        MatrixDense<TPrecision> rand_mat(MatrixDense<TPrecision>::Random(
            TestBase::bundle, m_rand, n_rand
        ));
        MatrixDense<TPrecision> rand_mat_trans(rand_mat.transpose());

        Vector<TPrecision> trans_mat_0(rand_mat_trans.get_col(0).copy_to_vec());
        Vector<TPrecision> trans_mat_1(rand_mat_trans.get_col(1).copy_to_vec());
        Vector<TPrecision> trans_mat_2(rand_mat_trans.get_col(2).copy_to_vec());

        ASSERT_VECTOR_NEAR(
            rand_mat.transpose_prod(
                Vector<TPrecision>(
                    TestBase::bundle,
                    {static_cast<TPrecision>(1), static_cast<TPrecision>(0),
                     static_cast<TPrecision>(0)}
                )
            ),
            trans_mat_0,
            trans_mat_0.abs()*static_cast<TPrecision>(Tol<TPrecision>::gamma(2))
        );
        ASSERT_VECTOR_NEAR(
            rand_mat.transpose_prod(
                Vector<TPrecision>(
                    TestBase::bundle,
                    {static_cast<TPrecision>(0), static_cast<TPrecision>(1),
                     static_cast<TPrecision>(0)}
                )
            ),
            trans_mat_1,
            trans_mat_1.abs()*Tol<TPrecision>::gamma_T(2)
        );
        ASSERT_VECTOR_NEAR(
            rand_mat.transpose_prod(
                Vector<TPrecision>(
                    TestBase::bundle,
                    {static_cast<TPrecision>(0), static_cast<TPrecision>(0),
                     static_cast<TPrecision>(1)}
                )
            ),
            trans_mat_2,
            trans_mat_2.abs()*Tol<TPrecision>::gamma_T(2)
        );
        ASSERT_VECTOR_NEAR(
            rand_mat.transpose_prod(
                Vector<TPrecision>(
                    TestBase::bundle,
                    {static_cast<TPrecision>(1), static_cast<TPrecision>(0.1),
                     static_cast<TPrecision>(0.01)}
                )
            ),
            (
                trans_mat_0*static_cast<TPrecision>(1) +
                trans_mat_1*static_cast<TPrecision>(0.1) +
                trans_mat_2*static_cast<TPrecision>(0.01)
            ),
            (
                trans_mat_0.abs()*static_cast<TPrecision>(1) +
                trans_mat_1.abs()*static_cast<TPrecision>(0.1) +
                trans_mat_2.abs()*static_cast<TPrecision>(0.01)
            )*Tol<TPrecision>::gamma_T(2)
        );

    }

    template <typename TPrecision>
    void TestSubsetcolsTransposeMatVec() {

        // Test manually
        MatrixDense<TPrecision> mat(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(2), static_cast<TPrecision>(3), static_cast<TPrecision>(-4)},
             {static_cast<TPrecision>(-5), static_cast<TPrecision>(-6), static_cast<TPrecision>(-7), static_cast<TPrecision>(-8)},
             {static_cast<TPrecision>(-9), static_cast<TPrecision>(1), static_cast<TPrecision>(2), static_cast<TPrecision>(3)}}
        );
        MatrixDense<TPrecision> trans_mat = mat.transpose();
        Vector<TPrecision> mat_r0(trans_mat.get_col(0).copy_to_vec());
        Vector<TPrecision> mat_r1(trans_mat.get_col(1).copy_to_vec());
        Vector<TPrecision> mat_r2(trans_mat.get_col(2).copy_to_vec());

        Vector<TPrecision> vec_3_1_0_0(
            TestBase::bundle,
            {static_cast<TPrecision>(1), static_cast<TPrecision>(0),
             static_cast<TPrecision>(0)}
        );
        Vector<TPrecision> vec_3_0_1_0(
            TestBase::bundle,
            {static_cast<TPrecision>(0), static_cast<TPrecision>(1),
             static_cast<TPrecision>(0)}
        );
        Vector<TPrecision> vec_3_0_0_1(
            TestBase::bundle,
            {static_cast<TPrecision>(0), static_cast<TPrecision>(0),
             static_cast<TPrecision>(1)}
        );
        Vector<TPrecision> vec_3_001_01_1(
            TestBase::bundle,
            {static_cast<TPrecision>(0.01), static_cast<TPrecision>(0.1),
             static_cast<TPrecision>(1.)}
        );

        // Test multiplication of first 2 columns
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod_subset_cols(0, 2, vec_3_1_0_0),
            mat_r0.get_slice(0, 2),
            mat_r0.get_slice(0, 2).abs()*Tol<TPrecision>::gamma_T(3)
        );
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod_subset_cols(0, 2, vec_3_0_1_0),
            mat_r1.get_slice(0, 2),
            mat_r1.get_slice(0, 2).abs()*Tol<TPrecision>::gamma_T(3)
        );
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod_subset_cols(0, 2, vec_3_0_0_1),
            mat_r2.get_slice(0, 2),
            mat_r2.get_slice(0, 2).abs()*Tol<TPrecision>::gamma_T(3)
        );

        ASSERT_VECTOR_NEAR(
            mat.transpose_prod_subset_cols(0, 2, vec_3_001_01_1),
            (mat_r0.get_slice(0, 2) *
             Scalar<TPrecision>(static_cast<TPrecision>(0.01))) +
            (mat_r1.get_slice(0, 2) *
             Scalar<TPrecision>(static_cast<TPrecision>(0.1))) +
            (mat_r2.get_slice(0, 2) *
             Scalar<TPrecision>(static_cast<TPrecision>(1.))),
            (
                (mat_r0.get_slice(0, 2).abs() *
                 Scalar<TPrecision>(static_cast<TPrecision>(0.01))) +
                (mat_r1.get_slice(0, 2).abs() *
                 Scalar<TPrecision>(static_cast<TPrecision>(0.1))) +
                (mat_r2.get_slice(0, 2).abs() *
                 Scalar<TPrecision>(static_cast<TPrecision>(1.)))
            )*Tol<TPrecision>::gamma_T(3)
        );

        // Test multiplication of last 2 columns
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod_subset_cols(2, 2, vec_3_1_0_0),
            mat_r0.get_slice(2, 2),
            mat_r0.get_slice(2, 2).abs()*Tol<TPrecision>::gamma_T(3)
        );
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod_subset_cols(2, 2, vec_3_0_1_0),
            mat_r1.get_slice(2, 2),
            mat_r1.get_slice(2, 2).abs()*Tol<TPrecision>::gamma_T(3)
        );
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod_subset_cols(2, 2, vec_3_0_0_1),
            mat_r2.get_slice(2, 2),
            mat_r2.get_slice(2, 2).abs()*Tol<TPrecision>::gamma_T(3)
        );

        ASSERT_VECTOR_NEAR(
            mat.transpose_prod_subset_cols(2, 2, vec_3_001_01_1),
            (mat_r0.get_slice(2, 2) *
             Scalar<TPrecision>(static_cast<TPrecision>(0.01))) +
            (mat_r1.get_slice(2, 2) *
             Scalar<TPrecision>(static_cast<TPrecision>(0.1))) +
            (mat_r2.get_slice(2, 2) *
             Scalar<TPrecision>(static_cast<TPrecision>(1.))),
            Tol<TPrecision>::gamma_T(3)
        );

        // Test multiplication of all columns
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod_subset_cols(0, 4, vec_3_1_0_0),
            mat_r0.get_slice(0, 4),
            mat_r0.abs()*Tol<TPrecision>::gamma_T(3)
        );
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod_subset_cols(0, 4, vec_3_0_1_0),
            mat_r1.get_slice(0, 4),
            mat_r1.abs()*Tol<TPrecision>::gamma_T(3)
        );
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod_subset_cols(0, 4, vec_3_0_0_1),
            mat_r2.get_slice(0, 4),
            mat_r2.abs()*Tol<TPrecision>::gamma_T(3)
        );
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod_subset_cols(0, 4, vec_3_001_01_1),
            (mat_r0.get_slice(0, 4) *
             Scalar<TPrecision>(static_cast<TPrecision>(0.01))) +
            (mat_r1.get_slice(0, 4) *
             Scalar<TPrecision>(static_cast<TPrecision>(0.1))) +
            (mat_r2.get_slice(0, 4) *
             Scalar<TPrecision>(static_cast<TPrecision>(1.))),
            (
                (mat_r0.get_slice(0, 4).abs() *
                 Scalar<TPrecision>(static_cast<TPrecision>(0.01))) +
                (mat_r1.get_slice(0, 4).abs() *
                 Scalar<TPrecision>(static_cast<TPrecision>(0.1))) +
                (mat_r2.get_slice(0, 4).abs() *
                 Scalar<TPrecision>(static_cast<TPrecision>(1.)))
            )*Tol<TPrecision>::gamma_T(3)
        );

        // Test multiplication of individual
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod_subset_cols(0, 1, vec_3_001_01_1),
            (mat_r0.get_slice(0, 1) *
             Scalar<TPrecision>(static_cast<TPrecision>(0.01))) +
            (mat_r1.get_slice(0, 1) *
             Scalar<TPrecision>(static_cast<TPrecision>(0.1))) +
            (mat_r2.get_slice(0, 1) *
             Scalar<TPrecision>(static_cast<TPrecision>(1.))),
            (
                (mat_r0.get_slice(0, 1).abs() *
                 Scalar<TPrecision>(static_cast<TPrecision>(0.01))) +
                (mat_r1.get_slice(0, 1).abs() *
                 Scalar<TPrecision>(static_cast<TPrecision>(0.1))) +
                (mat_r2.get_slice(0, 1).abs() *
                 Scalar<TPrecision>(static_cast<TPrecision>(1.)))
            )*Tol<TPrecision>::gamma_T(3)
        );
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod_subset_cols(1, 1, vec_3_001_01_1),
            (mat_r0.get_slice(1, 1) *
             Scalar<TPrecision>(static_cast<TPrecision>(0.01))) +
            (mat_r1.get_slice(1, 1) *
             Scalar<TPrecision>(static_cast<TPrecision>(0.1))) +
            (mat_r2.get_slice(1, 1) *
             Scalar<TPrecision>(static_cast<TPrecision>(1.))),
            (
                (mat_r0.get_slice(1, 1).abs() *
                 Scalar<TPrecision>(static_cast<TPrecision>(0.01))) +
                (mat_r1.get_slice(1, 1).abs() *
                 Scalar<TPrecision>(static_cast<TPrecision>(0.1))) +
                (mat_r2.get_slice(1, 1).abs() *
                 Scalar<TPrecision>(static_cast<TPrecision>(1.)))
            )*Tol<TPrecision>::gamma_T(3)
        );
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod_subset_cols(2, 1, vec_3_001_01_1),
            (mat_r0.get_slice(2, 1) *
             Scalar<TPrecision>(static_cast<TPrecision>(0.01))) +
            (mat_r1.get_slice(2, 1) *
             Scalar<TPrecision>(static_cast<TPrecision>(0.1))) +
            mat_r2.get_slice(2, 1),
            (
                (mat_r0.get_slice(2, 1).abs() *
                 Scalar<TPrecision>(static_cast<TPrecision>(0.01))) +
                (mat_r1.get_slice(2, 1).abs() *
                 Scalar<TPrecision>(static_cast<TPrecision>(0.1))) +
                (mat_r2.get_slice(2, 1).abs() *
                 Scalar<TPrecision>(static_cast<TPrecision>(1.)))
            )*Tol<TPrecision>::gamma_T(3)
        );
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod_subset_cols(3, 1, vec_3_001_01_1),
            (mat_r0.get_slice(3, 1) *
             Scalar<TPrecision>(static_cast<TPrecision>(0.01))) +
            (mat_r1.get_slice(3, 1) *
             Scalar<TPrecision>(static_cast<TPrecision>(0.1))) +
            (mat_r2.get_slice(3, 1) *
             Scalar<TPrecision>(static_cast<TPrecision>(1.))),
            (
                (mat_r0.get_slice(3, 1).abs() *
                 Scalar<TPrecision>(static_cast<TPrecision>(0.01))) +
                (mat_r1.get_slice(3, 1).abs() *
                 Scalar<TPrecision>(static_cast<TPrecision>(0.1))) +
                (mat_r2.get_slice(3, 1).abs() *
                 Scalar<TPrecision>(static_cast<TPrecision>(1.)))
            )*Tol<TPrecision>::gamma_T(3)
        );

    }

    template <typename TPrecision>
    void TestRandomSubsetcolsTransposeMatVec() {

        // Test manually
        MatrixDense<TPrecision> mat(MatrixDense<TPrecision>::Random(
            TestBase::bundle, 3, 4
        ));
        MatrixDense<TPrecision> trans_mat = mat.transpose();
        Vector<TPrecision> mat_r0(trans_mat.get_col(0).copy_to_vec());
        Vector<TPrecision> mat_r1(trans_mat.get_col(1).copy_to_vec());
        Vector<TPrecision> mat_r2(trans_mat.get_col(2).copy_to_vec());

        Vector<TPrecision> vec_3_1_0_0(
            TestBase::bundle,
            {static_cast<TPrecision>(1), static_cast<TPrecision>(0),
             static_cast<TPrecision>(0)}
        );
        Vector<TPrecision> vec_3_0_1_0(
            TestBase::bundle,
            {static_cast<TPrecision>(0), static_cast<TPrecision>(1),
             static_cast<TPrecision>(0)}
        );
        Vector<TPrecision> vec_3_0_0_1(
            TestBase::bundle,
            {static_cast<TPrecision>(0), static_cast<TPrecision>(0),
             static_cast<TPrecision>(1)}
        );
        Vector<TPrecision> vec_3_001_01_1(
            TestBase::bundle,
            {static_cast<TPrecision>(0.01), static_cast<TPrecision>(0.1),
             static_cast<TPrecision>(1.)}
        );

        // Test multiplication of first 2 columns
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod_subset_cols(0, 2, vec_3_1_0_0),
            mat_r0.get_slice(0, 2),
            mat_r0.get_slice(0, 2).abs()*Tol<TPrecision>::gamma_T(3)
        );
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod_subset_cols(0, 2, vec_3_0_1_0),
            mat_r1.get_slice(0, 2),
            mat_r1.get_slice(0, 2).abs()*Tol<TPrecision>::gamma_T(3)
        );
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod_subset_cols(0, 2, vec_3_0_0_1),
            mat_r2.get_slice(0, 2),
            mat_r2.get_slice(0, 2).abs()*Tol<TPrecision>::gamma_T(3)
        );
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod_subset_cols(0, 2, vec_3_001_01_1),
            (mat_r0.get_slice(0, 2) *
             Scalar<TPrecision>(static_cast<TPrecision>(0.01))) +
            (mat_r1.get_slice(0, 2) *
             Scalar<TPrecision>(static_cast<TPrecision>(0.1))) +
            (mat_r2.get_slice(0, 2) *
             Scalar<TPrecision>(static_cast<TPrecision>(1.))),
            (
                (mat_r0.get_slice(0, 2).abs() *
                 Scalar<TPrecision>(static_cast<TPrecision>(0.01))) +
                (mat_r1.get_slice(0, 2).abs() *
                 Scalar<TPrecision>(static_cast<TPrecision>(0.1))) +
                (mat_r2.get_slice(0, 2).abs() *
                 Scalar<TPrecision>(static_cast<TPrecision>(1.)))
            )*Tol<TPrecision>::gamma_T(3)
        );

        // Test multiplication of last 2 columns
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod_subset_cols(2, 2, vec_3_1_0_0),
            mat_r0.get_slice(2, 2),
            mat_r0.get_slice(2, 2).abs()*Tol<TPrecision>::gamma_T(3)
        );
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod_subset_cols(2, 2, vec_3_0_1_0),
            mat_r1.get_slice(2, 2),
            mat_r1.get_slice(2, 2).abs()*Tol<TPrecision>::gamma_T(3)
        );
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod_subset_cols(2, 2, vec_3_0_0_1),
            mat_r2.get_slice(2, 2),
            mat_r2.get_slice(2, 2).abs()*Tol<TPrecision>::gamma_T(3)
        );
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod_subset_cols(2, 2, vec_3_001_01_1),
            (mat_r0.get_slice(2, 2) *
             Scalar<TPrecision>(static_cast<TPrecision>(0.01))) +
            (mat_r1.get_slice(2, 2) *
             Scalar<TPrecision>(static_cast<TPrecision>(0.1))) +
            (mat_r2.get_slice(2, 2) *
             Scalar<TPrecision>(static_cast<TPrecision>(1.))),
            (
                (mat_r0.get_slice(2, 2).abs() *
                 Scalar<TPrecision>(static_cast<TPrecision>(0.01))) +
                (mat_r1.get_slice(2, 2).abs() *
                 Scalar<TPrecision>(static_cast<TPrecision>(0.1))) +
                (mat_r2.get_slice(2, 2).abs() *
                 Scalar<TPrecision>(static_cast<TPrecision>(1.)))
            )*Tol<TPrecision>::gamma_T(3)
        );

        // Test multiplication of all columns
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod_subset_cols(0, 4, vec_3_1_0_0),
            mat_r0.get_slice(0, 4),
            mat_r0.get_slice(0, 4).abs()*Tol<TPrecision>::gamma_T(3)
        );
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod_subset_cols(0, 4, vec_3_0_1_0),
            mat_r1.get_slice(0, 4),
            mat_r1.get_slice(0, 4).abs()*Tol<TPrecision>::gamma_T(3)
        );
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod_subset_cols(0, 4, vec_3_0_0_1),
            mat_r2.get_slice(0, 4),
            mat_r2.get_slice(0, 4).abs()*Tol<TPrecision>::gamma_T(3)
        );
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod_subset_cols(0, 4, vec_3_001_01_1),
            (mat_r0.get_slice(0, 4) *
             Scalar<TPrecision>(static_cast<TPrecision>(0.01))) +
            (mat_r1.get_slice(0, 4) *
             Scalar<TPrecision>(static_cast<TPrecision>(0.1))) +
            (mat_r2.get_slice(0, 4) *
             Scalar<TPrecision>(static_cast<TPrecision>(1.))) ,
            (
                (mat_r0.get_slice(0, 4).abs() *
                 Scalar<TPrecision>(static_cast<TPrecision>(0.01))) +
                (mat_r1.get_slice(0, 4).abs() *
                 Scalar<TPrecision>(static_cast<TPrecision>(0.1))) +
                (mat_r2.get_slice(0, 4).abs() *
                 Scalar<TPrecision>(static_cast<TPrecision>(1.)))
            )*Tol<TPrecision>::gamma_T(3)
        );

        // Test multiplication of individual
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod_subset_cols(0, 1, vec_3_001_01_1),
            (mat_r0.get_slice(0, 1) *
             Scalar<TPrecision>(static_cast<TPrecision>(0.01))) +
            (mat_r1.get_slice(0, 1) *
             Scalar<TPrecision>(static_cast<TPrecision>(0.1))) +
            (mat_r2.get_slice(0, 1) *
             Scalar<TPrecision>(static_cast<TPrecision>(1.))),
            (
                (mat_r0.get_slice(0, 1).abs() *
                 Scalar<TPrecision>(static_cast<TPrecision>(0.01))) +
                (mat_r1.get_slice(0, 1).abs() *
                 Scalar<TPrecision>(static_cast<TPrecision>(0.1))) +
                (mat_r2.get_slice(0, 1).abs() *
                 Scalar<TPrecision>(static_cast<TPrecision>(1.)))
            )*Tol<TPrecision>::gamma_T(3)
        );
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod_subset_cols(1, 1, vec_3_001_01_1),
            (mat_r0.get_slice(1, 1) *
             Scalar<TPrecision>(static_cast<TPrecision>(0.01))) +
            (mat_r1.get_slice(1, 1) *
             Scalar<TPrecision>(static_cast<TPrecision>(0.1))) +
            (mat_r2.get_slice(1, 1) *
             Scalar<TPrecision>(static_cast<TPrecision>(1.))),
            (
                (mat_r0.get_slice(1, 1).abs() *
                 Scalar<TPrecision>(static_cast<TPrecision>(0.01))) +
                (mat_r1.get_slice(1, 1).abs() *
                 Scalar<TPrecision>(static_cast<TPrecision>(0.1))) +
                (mat_r2.get_slice(1, 1).abs() *
                 Scalar<TPrecision>(static_cast<TPrecision>(1.)))
            )*Tol<TPrecision>::gamma_T(3)
        );
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod_subset_cols(2, 1, vec_3_001_01_1),
            (mat_r0.get_slice(2, 1) *
             Scalar<TPrecision>(static_cast<TPrecision>(0.01))) +
            (mat_r1.get_slice(2, 1) *
             Scalar<TPrecision>(static_cast<TPrecision>(0.1))) +
            (mat_r2.get_slice(2, 1) *
             Scalar<TPrecision>(static_cast<TPrecision>(1.))),
            (
                (mat_r0.get_slice(2, 1).abs() *
                 Scalar<TPrecision>(static_cast<TPrecision>(0.01))) +
                (mat_r1.get_slice(2, 1).abs() *
                 Scalar<TPrecision>(static_cast<TPrecision>(0.1))) +
                (mat_r2.get_slice(2, 1).abs() *
                 Scalar<TPrecision>(static_cast<TPrecision>(1.)))
            )*Tol<TPrecision>::gamma_T(3)
        );
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod_subset_cols(3, 1, vec_3_001_01_1),
            (mat_r0.get_slice(3, 1) *
             Scalar<TPrecision>(static_cast<TPrecision>(0.01))) +
            (mat_r1.get_slice(3, 1) *
             Scalar<TPrecision>(static_cast<TPrecision>(0.1))) +
            (mat_r2.get_slice(3, 1) *
             Scalar<TPrecision>(static_cast<TPrecision>(1.))),
            (
                (mat_r0.get_slice(3, 1).abs() *
                 Scalar<TPrecision>(static_cast<TPrecision>(0.01))) +
                (mat_r1.get_slice(3, 1).abs() *
                 Scalar<TPrecision>(static_cast<TPrecision>(0.1))) +
                (mat_r2.get_slice(3, 1).abs() *
                 Scalar<TPrecision>(static_cast<TPrecision>(1.)))
            )*Tol<TPrecision>::gamma_T(3)
        );

    }

    template <typename TPrecision>
    void TestLimitRandomSubsetcolsTransposeMatVec() {

        const int m(12);
        const int n(7);

        MatrixDense<TPrecision> mat(MatrixDense<TPrecision>::Random(
            TestBase::bundle, m, n
        ));
        Vector<TPrecision> vec(Vector<TPrecision>::Random(
            TestBase::bundle, m
        ));

        ASSERT_VECTOR_EQ(
            mat.transpose_prod_subset_cols(0, n, vec),
            mat.transpose_prod(vec)
        );

    }

    template <typename TPrecision>
    void TestBadSubsetcolsTransposeMatVec() {

        MatrixDense<TPrecision> mat(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(2),
              static_cast<TPrecision>(3), static_cast<TPrecision>(4)},
             {static_cast<TPrecision>(5), static_cast<TPrecision>(6),
              static_cast<TPrecision>(7), static_cast<TPrecision>(8)},
             {static_cast<TPrecision>(9), static_cast<TPrecision>(10),
              static_cast<TPrecision>(11), static_cast<TPrecision>(12)}}
        );

        Vector<TPrecision> valid_vec(
            TestBase::bundle,
            {static_cast<TPrecision>(1), static_cast<TPrecision>(1),
             static_cast<TPrecision>(1)}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() { mat.transpose_prod_subset_cols(-1, 3, valid_vec); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() { mat.transpose_prod_subset_cols(2, 3, valid_vec); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() { mat.transpose_prod_subset_cols(3, 2, valid_vec); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() { mat.transpose_prod_subset_cols(1, -1, valid_vec); }
        );

        Vector<TPrecision> vec_too_small(
            TestBase::bundle,
            {static_cast<TPrecision>(1), static_cast<TPrecision>(1)}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() { mat.transpose_prod_subset_cols(0, 2, vec_too_small); }
        );

        Vector<TPrecision> vec_too_large(
            TestBase::bundle, 
            {static_cast<TPrecision>(1), static_cast<TPrecision>(1),
             static_cast<TPrecision>(1), static_cast<TPrecision>(1)}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() { mat.transpose_prod_subset_cols(0, 4, vec_too_large); }
        );
    
    }

    template <typename TPrecision>
    void TestRandomTranspose() {

        constexpr int m_rand(4);
        constexpr int n_rand(3);
        MatrixDense<TPrecision> mat(MatrixDense<TPrecision>::Random(
            TestBase::bundle, n_rand, m_rand
        ));

        MatrixDense<TPrecision> mat_transposed(mat.transpose());
        ASSERT_EQ(mat_transposed.rows(), m_rand);
        ASSERT_EQ(mat_transposed.cols(), n_rand);
        for (int i=0; i<m_rand; ++i) {
            for (int j=0; j<n_rand; ++j) {
                ASSERT_EQ(
                    mat_transposed.get_elem(i, j).get_scalar(),
                    mat.get_elem(j, i).get_scalar()
                );
            }
        }

    }

    template <typename TPrecision>
    void TestAbs() {

        // Test manually
        constexpr int m_manual(4);
        constexpr int n_manual(3);
        MatrixDense<TPrecision> mat(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(-2),
              static_cast<TPrecision>(3), static_cast<TPrecision>(-4)},
             {static_cast<TPrecision>(-5), static_cast<TPrecision>(-6),
              static_cast<TPrecision>(-7), static_cast<TPrecision>(-8)},
             {static_cast<TPrecision>(9), static_cast<TPrecision>(10),
              static_cast<TPrecision>(11), static_cast<TPrecision>(12)}}
        );
        MatrixDense<TPrecision> test(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(2),
              static_cast<TPrecision>(3), static_cast<TPrecision>(4)},
             {static_cast<TPrecision>(5), static_cast<TPrecision>(6),
              static_cast<TPrecision>(7), static_cast<TPrecision>(8)},
             {static_cast<TPrecision>(9), static_cast<TPrecision>(10),
              static_cast<TPrecision>(11), static_cast<TPrecision>(12)}}
        );
        ASSERT_MATRIX_EQ(mat.abs(), test);

    }

    template <typename TPrecision>
    void TestRandomAbs() {

        constexpr int m_rand(4);
        constexpr int n_rand(3);
        MatrixDense<TPrecision> mat(MatrixDense<TPrecision>::Random(
            TestBase::bundle, m_rand, n_rand
        ));

        MatrixDense<TPrecision> test_mat = mat.abs();

        for (int i=0; i<m_rand; ++i) {
            for (int j=0; j<n_rand; ++j) {
                Scalar<TPrecision> abs_elem(mat.get_elem(i, j).get_scalar());
                ASSERT_EQ(
                    test_mat.get_elem(i, j).get_scalar(),
                    abs_elem.abs().get_scalar()
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

TEST_F(MatrixDense_Test, TestRandomUTMatrixCreation) {
    TestRandomUTMatrixCreation<__half>();
    TestRandomUTMatrixCreation<float>();
    TestRandomUTMatrixCreation<double>();
}

TEST_F(MatrixDense_Test, TestRandomLTMatrixCreation) {
    TestRandomLTMatrixCreation<__half>();
    TestRandomLTMatrixCreation<float>();
    TestRandomLTMatrixCreation<double>();
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

TEST_F(MatrixDense_Test, TestRandomMatVec) {
    TestRandomMatVec<__half>();
    TestRandomMatVec<float>();
    TestRandomMatVec<double>();
}

TEST_F(MatrixDense_Test, TestBadMatVec) {
    TestBadMatVec<__half>();
    TestBadMatVec<float>();
    TestBadMatVec<double>();
}

TEST_F(MatrixDense_Test, TestSubsetcolsMatVec) {
    TestSubsetcolsMatVec<__half>();
    TestSubsetcolsMatVec<float>();
    TestSubsetcolsMatVec<double>();
}

TEST_F(MatrixDense_Test, TestRandomSubsetcolsMatVec) {
    TestRandomSubsetcolsMatVec<__half>();
    TestRandomSubsetcolsMatVec<float>();
    TestRandomSubsetcolsMatVec<double>();
}

TEST_F(MatrixDense_Test, TestLimitRandomSubsetcolsMatVec) {
    TestLimitRandomSubsetcolsMatVec<__half>();
    TestLimitRandomSubsetcolsMatVec<float>();
    TestLimitRandomSubsetcolsMatVec<double>();
}


TEST_F(MatrixDense_Test, TestBadSubsetcolsMatVec) {
    TestBadSubsetcolsMatVec<__half>();
    TestBadSubsetcolsMatVec<float>();
    TestBadSubsetcolsMatVec<double>();
}

TEST_F(MatrixDense_Test, TestTransposeMatVec) {
    TestTransposeMatVec<__half>();
    TestTransposeMatVec<float>();
    TestTransposeMatVec<double>();
}

TEST_F(MatrixDense_Test, TestRandomTransposeMatVec) {
    TestRandomTransposeMatVec<__half>();
    TestRandomTransposeMatVec<float>();
    TestRandomTransposeMatVec<double>();
}

TEST_F(MatrixDense_Test, TestBadTransposeMatVec) {
    TestBadTransposeMatVec<__half>();
    TestBadTransposeMatVec<float>();
    TestBadTransposeMatVec<double>();
}

TEST_F(MatrixDense_Test, TestSubsetcolsTransposeMatVec) {
    TestSubsetcolsTransposeMatVec<__half>();
    TestSubsetcolsTransposeMatVec<float>();
    TestSubsetcolsTransposeMatVec<double>();
}

TEST_F(MatrixDense_Test, TestRandomSubsetcolsTransposeMatVec) {
    TestRandomSubsetcolsTransposeMatVec<__half>();
    TestRandomSubsetcolsTransposeMatVec<float>();
    TestRandomSubsetcolsTransposeMatVec<double>();
}

TEST_F(MatrixDense_Test, TestLimitRandomSubsetcolsTransposeMatVec) {
    TestLimitRandomSubsetcolsTransposeMatVec<__half>();
    TestLimitRandomSubsetcolsTransposeMatVec<float>();
    TestLimitRandomSubsetcolsTransposeMatVec<double>();
}

TEST_F(MatrixDense_Test, TestBadSubsetcolsTransposeMatVec) {
    TestBadSubsetcolsTransposeMatVec<__half>();
    TestBadSubsetcolsTransposeMatVec<float>();
    TestBadSubsetcolsTransposeMatVec<double>();
}

TEST_F(MatrixDense_Test, TestTranspose) {
    TestTranspose<__half>();
    TestTranspose<float>();
    TestTranspose<double>();
}

TEST_F(MatrixDense_Test, TestRandomTranspose) {
    TestRandomTranspose<__half>();
    TestRandomTranspose<float>();
    TestRandomTranspose<double>();
}

TEST_F(MatrixDense_Test, TestAbs) {
    TestAbs<__half>();
    TestAbs<float>();
    TestAbs<double>();
}

TEST_F(MatrixDense_Test, TestRandomAbs) {
    TestRandomAbs<__half>();
    TestRandomAbs<float>();
    TestRandomAbs<double>();
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

class MatrixDense_Substitution_Test:
    public Matrix_Substitution_Test<MatrixDense>
{};

TEST_F(MatrixDense_Substitution_Test, TestForwardSubstitution) {
    TestForwardSubstitution<__half>();
    TestForwardSubstitution<float>();
    TestForwardSubstitution<double>();
}

TEST_F(MatrixDense_Substitution_Test, TestRandomForwardSubstitution) {
    TestRandomForwardSubstitution<__half>();
    TestRandomForwardSubstitution<float>();
    TestRandomForwardSubstitution<double>();
}

TEST_F(MatrixDense_Substitution_Test, TestRandomSparseForwardSubstitution) {
    TestRandomSparseForwardSubstitution<__half>();
    TestRandomSparseForwardSubstitution<float>();
    TestRandomSparseForwardSubstitution<double>();
}

TEST_F(MatrixDense_Substitution_Test, TestBackwardSubstitution) {
    TestBackwardSubstitution<__half>();
    TestBackwardSubstitution<float>();
    TestBackwardSubstitution<double>();
}

TEST_F(MatrixDense_Substitution_Test, TestRandomBackwardSubstitution) {
    TestRandomBackwardSubstitution<__half>();
    TestRandomBackwardSubstitution<float>();
    TestRandomBackwardSubstitution<double>();
}

TEST_F(MatrixDense_Substitution_Test, TestRandomSparseBackwardSubstitution) {
    TestRandomSparseBackwardSubstitution<__half>();
    TestRandomSparseBackwardSubstitution<float>();
    TestRandomSparseBackwardSubstitution<double>();
}