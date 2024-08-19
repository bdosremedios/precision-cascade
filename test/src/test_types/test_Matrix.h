#include "test.h"

#include "types/GeneralMatrix/GeneralMatrix_gpu_constants.cuh"

template <template <typename> typename TMatrix>
class Matrix_Test: public TestBase
{
protected:

    template <typename TPrecision>
    void TestCoeffAccess() {
        
        constexpr int m(24);
        constexpr int n(12);
        TMatrix<TPrecision> test_mat(TestBase::bundle, m, n);

        Scalar<TPrecision> elem(static_cast<TPrecision>(1));
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                test_mat.set_elem(i, j, elem);
                elem += Scalar<TPrecision>(static_cast<TPrecision>(1));
            }
        }

        TPrecision test_elem = static_cast<TPrecision>(1);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_EQ(
                    test_mat.get_elem(i, j).get_scalar(),
                    test_elem
                );
                test_elem += static_cast<TPrecision>(1);
            }
        }

        // Set index 0 row as all -1
        for (int j=0; j<n; ++j) {
            test_mat.set_elem(
                0, j, Scalar<TPrecision>(static_cast<TPrecision>(-1.))
            );
        }

        // Test matches modified matrix
        test_elem = static_cast<TPrecision>(1);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                if (i == 0) {
                    ASSERT_EQ(
                        test_mat.get_elem(i, j).get_scalar(),
                        static_cast<TPrecision>(-1.)
                    );
                } else {
                    ASSERT_EQ(
                        test_mat.get_elem(i, j).get_scalar(),
                        test_elem
                    );
                }
                test_elem += static_cast<TPrecision>(1);
            }
        }

        // Set index 4 row as decreasing by -1 from -1
        Scalar<TPrecision> row_5_elem(static_cast<TPrecision>(-1.));
        for (int j=0; j<n; ++j) {
            test_mat.set_elem(4, j, row_5_elem);
            row_5_elem += Scalar<TPrecision>(static_cast<TPrecision>(-1.));
        }

        // Test matches modified matrix
        test_elem = static_cast<TPrecision>(1);
        TPrecision row_5_test_elem = static_cast<TPrecision>(-1.);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                if (i == 0) {
                    ASSERT_EQ(
                        test_mat.get_elem(i, j).get_scalar(),
                        static_cast<TPrecision>(-1.)
                    );
                } else if (i == 4) {
                    ASSERT_EQ(
                        test_mat.get_elem(i, j).get_scalar(),
                        row_5_test_elem
                    );
                    row_5_test_elem += static_cast<TPrecision>(-1.);
                } else {
                    ASSERT_EQ(
                        test_mat.get_elem(i, j).get_scalar(),
                        test_elem
                    );
                }
                test_elem += static_cast<TPrecision>(1);
            }
        }

        // Set index 2 col as increasing by 1 from -5
        Scalar<TPrecision> coL_3_elem(static_cast<TPrecision>(-5.));
        for (int i=0; i<m; ++i) {
            test_mat.set_elem(i, 2, coL_3_elem);
            coL_3_elem += Scalar<TPrecision>(static_cast<TPrecision>(1.));
        }

        // Test matches modified matrix
        test_elem = static_cast<TPrecision>(1);
        row_5_test_elem = static_cast<TPrecision>(-1.);
        TPrecision coL_3_test_elem = static_cast<TPrecision>(-5.);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                if (j == 2) {
                    ASSERT_EQ(
                        test_mat.get_elem(i, j).get_scalar(),
                        coL_3_test_elem
                    );
                    coL_3_test_elem += static_cast<TPrecision>(1.);
                } else if (i == 0) {
                    ASSERT_EQ(
                        test_mat.get_elem(i, j).get_scalar(),
                        static_cast<TPrecision>(-1.)
                    );
                } else if (i == 4) {
                    ASSERT_EQ(
                        test_mat.get_elem(i, j).get_scalar(),
                        row_5_test_elem
                    );
                    row_5_test_elem += static_cast<TPrecision>(-1.);
                    if (j == 1) { row_5_test_elem += static_cast<TPrecision>(-1.); }
                } else {
                    ASSERT_EQ(
                        test_mat.get_elem(i, j).get_scalar(),
                        test_elem
                    );
                }
                test_elem += static_cast<TPrecision>(1);
            }
        }

    }

    void TestBadCoeffAccess() {
        
        constexpr int m(24);
        constexpr int n(12);
        TMatrix<double> test_mat(TMatrix<double>::Random(
            TestBase::bundle, m, n
        ));

        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [&]() { test_mat.get_elem(0, -1); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [&]() { test_mat.get_elem(0, n); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [&]() { test_mat.get_elem(-1, 0); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [&]() { test_mat.get_elem(m, 0); }
        );

        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() mutable { test_mat.set_elem(0, -1, Scalar<double>(0.)); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() mutable { test_mat.set_elem(0, n, Scalar<double>(0.)); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() mutable { test_mat.set_elem(-1, 0, Scalar<double>(0.)); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() mutable { test_mat.set_elem(m, 0, Scalar<double>(0.)); }
        );

    }

    template <typename TPrecision>
    void TestPropertyAccess() {

        constexpr int m_1(62);
        constexpr int n_1(3);
        TMatrix<TPrecision> mat_1(TestBase::bundle, m_1, n_1);
        ASSERT_EQ(mat_1.rows(), m_1);
        ASSERT_EQ(mat_1.cols(), n_1);

        constexpr int m_2(15);
        constexpr int n_2(90);
        TMatrix<TPrecision> mat_2(TestBase::bundle, m_2, n_2);
        ASSERT_EQ(mat_2.rows(), m_2);
        ASSERT_EQ(mat_2.cols(), n_2);

        constexpr int m_3(20);
        constexpr int n_3(20);
        TMatrix<TPrecision> mat_3(TestBase::bundle, m_3, n_3);
        ASSERT_EQ(mat_3.rows(), m_3);
        ASSERT_EQ(mat_3.cols(), n_3);

    }

    template <typename TPrecision>
    void TestConstruction() {

        TMatrix<TPrecision> test_mat_empty(TestBase::bundle);
        ASSERT_EQ(test_mat_empty.rows(), 0);
        ASSERT_EQ(test_mat_empty.cols(), 0);

        constexpr int m(12);
        TMatrix<TPrecision> test_mat_square(TestBase::bundle, m, m);
        ASSERT_EQ(test_mat_square.rows(), m);
        ASSERT_EQ(test_mat_square.cols(), m);

        constexpr int n(33);
        TMatrix<TPrecision> test_mat_wide(TestBase::bundle, m, n);
        ASSERT_EQ(test_mat_wide.rows(), m);
        ASSERT_EQ(test_mat_wide.cols(), n);

        TMatrix<TPrecision> test_mat_tall(TestBase::bundle, n, m);
        ASSERT_EQ(test_mat_tall.rows(), n);
        ASSERT_EQ(test_mat_tall.cols(), m);

    }

    void TestBadConstruction() {

        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            []() { TMatrix<double> test_mat(TestBase::bundle, -1, 4); }
        );

        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            []() { TMatrix<double> test_mat(TestBase::bundle, 5, -2); }
        );

        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            []() { TMatrix<double> test_mat(TestBase::bundle, -1, -2); }
        );

    }

    template <typename TPrecision>
    void TestListInitialization() {

        TMatrix<TPrecision> test_mat_0_0 (TestBase::bundle, {});
        ASSERT_EQ(test_mat_0_0.rows(), 0);
        ASSERT_EQ(test_mat_0_0.cols(), 0);

        TMatrix<TPrecision> test_mat_0_1 (TestBase::bundle, {{}});
        ASSERT_EQ(test_mat_0_1.rows(), 1);
        ASSERT_EQ(test_mat_0_1.cols(), 0);

        TMatrix<TPrecision> test_mat_1(
            TestBase::bundle,
            {{static_cast<TPrecision>(5.), static_cast<TPrecision>(3.),
              static_cast<TPrecision>(27.)},
             {static_cast<TPrecision>(88.), static_cast<TPrecision>(-4.),
              static_cast<TPrecision>(-6.)},
             {static_cast<TPrecision>(100.), static_cast<TPrecision>(12.),
              static_cast<TPrecision>(2.)}}
        );
        ASSERT_EQ(test_mat_1.rows(), 3);
        ASSERT_EQ(test_mat_1.cols(), 3);
        ASSERT_EQ(
            test_mat_1.get_elem(0, 0).get_scalar(),
            static_cast<TPrecision>(5.)
        );
        ASSERT_EQ(
            test_mat_1.get_elem(0, 1).get_scalar(),
            static_cast<TPrecision>(3.)
        );
        ASSERT_EQ(
            test_mat_1.get_elem(0, 2).get_scalar(),
            static_cast<TPrecision>(27.)
        );
        ASSERT_EQ(
            test_mat_1.get_elem(1, 0).get_scalar(),
            static_cast<TPrecision>(88.)
        );
        ASSERT_EQ(
            test_mat_1.get_elem(1, 1).get_scalar(),
            static_cast<TPrecision>(-4.)
        );
        ASSERT_EQ(
            test_mat_1.get_elem(1, 2).get_scalar(),
            static_cast<TPrecision>(-6.)
        );
        ASSERT_EQ(
            test_mat_1.get_elem(2, 0).get_scalar(),
            static_cast<TPrecision>(100.)
        );
        ASSERT_EQ(
            test_mat_1.get_elem(2, 1).get_scalar(),
            static_cast<TPrecision>(12.)
        );
        ASSERT_EQ(
            test_mat_1.get_elem(2, 2).get_scalar(),
            static_cast<TPrecision>(2.)
        );

        TMatrix<TPrecision> test_mat_wide(
            TestBase::bundle,
            {{static_cast<TPrecision>(7.), static_cast<TPrecision>(5.),
              static_cast<TPrecision>(3.)},
             {static_cast<TPrecision>(1.), static_cast<TPrecision>(6.),
              static_cast<TPrecision>(2.)}}
        );
        ASSERT_EQ(test_mat_wide.rows(), 2);
        ASSERT_EQ(test_mat_wide.cols(), 3);
        ASSERT_EQ(
            test_mat_wide.get_elem(0, 0).get_scalar(),
            static_cast<TPrecision>(7.)
        );
        ASSERT_EQ(
            test_mat_wide.get_elem(0, 1).get_scalar(),
            static_cast<TPrecision>(5.)
        );
        ASSERT_EQ(
            test_mat_wide.get_elem(0, 2).get_scalar(),
            static_cast<TPrecision>(3.)
        );
        ASSERT_EQ(
            test_mat_wide.get_elem(1, 0).get_scalar(),
            static_cast<TPrecision>(1.)
        );
        ASSERT_EQ(
            test_mat_wide.get_elem(1, 1).get_scalar(),
            static_cast<TPrecision>(6.)
        );
        ASSERT_EQ(
            test_mat_wide.get_elem(1, 2).get_scalar(),
            static_cast<TPrecision>(2.)
        );
        
        TMatrix<TPrecision> test_mat_tall(
            TestBase::bundle,
            {{static_cast<TPrecision>(7.), static_cast<TPrecision>(5.)},
             {static_cast<TPrecision>(1.), static_cast<TPrecision>(6.)},
             {static_cast<TPrecision>(3.), static_cast<TPrecision>(2.)},
             {static_cast<TPrecision>(43.), static_cast<TPrecision>(9.)}}
        );
        ASSERT_EQ(test_mat_tall.rows(), 4);
        ASSERT_EQ(test_mat_tall.cols(), 2);
        ASSERT_EQ(
            test_mat_tall.get_elem(0, 0).get_scalar(),
            static_cast<TPrecision>(7.)
        );
        ASSERT_EQ(
            test_mat_tall.get_elem(0, 1).get_scalar(),
            static_cast<TPrecision>(5.)
        );
        ASSERT_EQ(
            test_mat_tall.get_elem(1, 0).get_scalar(),
            static_cast<TPrecision>(1.)
        );
        ASSERT_EQ(
            test_mat_tall.get_elem(1, 1).get_scalar(),
            static_cast<TPrecision>(6.)
        );
        ASSERT_EQ(
            test_mat_tall.get_elem(2, 0).get_scalar(),
            static_cast<TPrecision>(3.)
        );
        ASSERT_EQ(
            test_mat_tall.get_elem(2, 1).get_scalar(),
            static_cast<TPrecision>(2.)
        );
        ASSERT_EQ(
            test_mat_tall.get_elem(3, 0).get_scalar(),
            static_cast<TPrecision>(43.)
        );
        ASSERT_EQ(
            test_mat_tall.get_elem(3, 1).get_scalar(),
            static_cast<TPrecision>(9.)
        );

    }

    void TestBadListInitialization() {

        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            []() {
                TMatrix<double> mat(
                    TestBase::bundle, {{1, 2, 3, 4}, {1, 2, 3}}
                );
            }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            []() {
                TMatrix<double> mat(
                    TestBase::bundle, {{1, 2}, {1, 2, 3}}
                );
            }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            []() {
                TMatrix<double> mat(
                    TestBase::bundle, {{1, 2}, {}}
                );
            }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            []() {
                TMatrix<double> mat(
                    TestBase::bundle, {{}, {1, 2, 3}}
                );
            }
        );

    }

    template <typename TPrecision>
    void TestNonZeros() {

        TMatrix<TPrecision> test_non_zeros_some(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(0),
              static_cast<TPrecision>(3), static_cast<TPrecision>(0)},
             {static_cast<TPrecision>(0), static_cast<TPrecision>(7),
              static_cast<TPrecision>(0), static_cast<TPrecision>(9)},
             {static_cast<TPrecision>(11), static_cast<TPrecision>(12),
              static_cast<TPrecision>(13), static_cast<TPrecision>(14)},
             {static_cast<TPrecision>(16), static_cast<TPrecision>(0),
              static_cast<TPrecision>(18), static_cast<TPrecision>(0)}}
        );
        ASSERT_EQ(test_non_zeros_some.non_zeros(), 10);

        TMatrix<TPrecision> test_non_zeros_all(
            TestBase::bundle,
            {{static_cast<TPrecision>(0), static_cast<TPrecision>(0),
              static_cast<TPrecision>(0)},
             {static_cast<TPrecision>(0), static_cast<TPrecision>(0),
              static_cast<TPrecision>(0)},
             {static_cast<TPrecision>(0), static_cast<TPrecision>(0),
              static_cast<TPrecision>(0)},
             {static_cast<TPrecision>(0), static_cast<TPrecision>(0),
              static_cast<TPrecision>(0)}}
        );
        ASSERT_EQ(test_non_zeros_all.non_zeros(), 0);

        TMatrix<TPrecision> test_non_zeros_none(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(2),
              static_cast<TPrecision>(3), static_cast<TPrecision>(4),
              static_cast<TPrecision>(5)},
             {static_cast<TPrecision>(6), static_cast<TPrecision>(7),
              static_cast<TPrecision>(8), static_cast<TPrecision>(9),
              static_cast<TPrecision>(10)},
             {static_cast<TPrecision>(11), static_cast<TPrecision>(12),
              static_cast<TPrecision>(13), static_cast<TPrecision>(14),
              static_cast<TPrecision>(15)}}
        );
        ASSERT_EQ(test_non_zeros_none.non_zeros(), 15);

    }

    void TestPrintAndInfoString() {

        TMatrix<double> test_mat(
            TestBase::bundle,
            {{1., 2., 3.},
             {4., 0., 0.},
             {0., 8., 9.}}
        );
        std::cout << test_mat.get_matrix_string() << std::endl;
        std::cout << test_mat.get_info_string() << std::endl;
    
    }

    template <typename TPrecision>
    void TestCopyAssignment() {

        TMatrix<TPrecision> test_mat_empty(TestBase::bundle, {{}});
        TMatrix<TPrecision> test_mat_2_2(
            TestBase::bundle,
            {{static_cast<TPrecision>(-3.), static_cast<TPrecision>(0.)},
             {static_cast<TPrecision>(1.), static_cast<TPrecision>(10.)}}
        );
        TMatrix<TPrecision> test_mat_2_4(
            TestBase::bundle,
            {{static_cast<TPrecision>(0.), static_cast<TPrecision>(-1.),
              static_cast<TPrecision>(-2.), static_cast<TPrecision>(-2.)},
             {static_cast<TPrecision>(10.), static_cast<TPrecision>(0.),
              static_cast<TPrecision>(12.), static_cast<TPrecision>(12.)}}
        );
        TMatrix<TPrecision> test_mat_4_3(
            TestBase::bundle,
            {{static_cast<TPrecision>(1.), static_cast<TPrecision>(2.),
              static_cast<TPrecision>(3.)},
             {static_cast<TPrecision>(4.), static_cast<TPrecision>(5.),
              static_cast<TPrecision>(6.)},
             {static_cast<TPrecision>(7.), static_cast<TPrecision>(8.),
              static_cast<TPrecision>(9.)},
             {static_cast<TPrecision>(10.), static_cast<TPrecision>(11.),
              static_cast<TPrecision>(12.)}}
        );

        // Copy to empty
        test_mat_empty = test_mat_4_3;
        ASSERT_EQ(test_mat_empty.rows(), test_mat_4_3.rows());
        ASSERT_EQ(test_mat_empty.cols(), test_mat_4_3.cols());
        ASSERT_MATRIX_EQ(test_mat_empty, test_mat_4_3);

        // Copy to populated
        test_mat_2_2 = test_mat_4_3;
        ASSERT_EQ(test_mat_2_2.rows(), test_mat_4_3.rows());
        ASSERT_EQ(test_mat_2_2.cols(), test_mat_4_3.cols());
        ASSERT_MATRIX_EQ(test_mat_2_2, test_mat_4_3);

        // Reassignment
        test_mat_2_2 = test_mat_2_4;
        ASSERT_EQ(test_mat_2_2.rows(), test_mat_2_4.rows());
        ASSERT_EQ(test_mat_2_2.cols(), test_mat_2_4.cols());
        ASSERT_MATRIX_EQ(test_mat_2_2, test_mat_2_4);

        // Transitive assignment
        test_mat_empty = test_mat_2_2;
        ASSERT_EQ(test_mat_empty.rows(), test_mat_2_4.rows());
        ASSERT_EQ(test_mat_empty.cols(), test_mat_2_4.cols());
        ASSERT_MATRIX_EQ(test_mat_empty, test_mat_2_4);

        // Self-assignment
        test_mat_4_3 = test_mat_4_3;
        ASSERT_EQ(test_mat_4_3.rows(), 4);
        ASSERT_EQ(test_mat_4_3.cols(), 3);
        ASSERT_EQ(
            test_mat_4_3.get_elem(0, 0).get_scalar(),
            static_cast<TPrecision>(1.)
        );
        ASSERT_EQ(
            test_mat_4_3.get_elem(0, 1).get_scalar(),
            static_cast<TPrecision>(2.)
        );
        ASSERT_EQ(
            test_mat_4_3.get_elem(0, 2).get_scalar(),
            static_cast<TPrecision>(3.)
        );
        ASSERT_EQ(
            test_mat_4_3.get_elem(1, 0).get_scalar(),
            static_cast<TPrecision>(4.)
        );
        ASSERT_EQ(
            test_mat_4_3.get_elem(1, 1).get_scalar(),
            static_cast<TPrecision>(5.)
        );
        ASSERT_EQ(
            test_mat_4_3.get_elem(1, 2).get_scalar(),
            static_cast<TPrecision>(6.)
        );
        ASSERT_EQ(
            test_mat_4_3.get_elem(2, 0).get_scalar(),
            static_cast<TPrecision>(7.)
        );
        ASSERT_EQ(
            test_mat_4_3.get_elem(2, 1).get_scalar(),
            static_cast<TPrecision>(8.)
        );
        ASSERT_EQ(
            test_mat_4_3.get_elem(2, 2).get_scalar(),
            static_cast<TPrecision>(9.)
        );
        ASSERT_EQ(
            test_mat_4_3.get_elem(3, 0).get_scalar(),
            static_cast<TPrecision>(10.)
        );
        ASSERT_EQ(
            test_mat_4_3.get_elem(3, 1).get_scalar(),
            static_cast<TPrecision>(11.)
        );
        ASSERT_EQ(
            test_mat_4_3.get_elem(3, 2).get_scalar(),
            static_cast<TPrecision>(12.)
        );

    }

    template <typename TPrecision>
    void TestCopyConstructor() {

        TMatrix<TPrecision> test_mat_2_4(
            TestBase::bundle,
            {{static_cast<TPrecision>(0.), static_cast<TPrecision>(-1.),
              static_cast<TPrecision>(-2.), static_cast<TPrecision>(-2.)},
             {static_cast<TPrecision>(10.), static_cast<TPrecision>(0.),
              static_cast<TPrecision>(12.), static_cast<TPrecision>(12.)}}
        );

        TMatrix<TPrecision> test_mat_copied(test_mat_2_4);
        ASSERT_EQ(test_mat_copied.rows(), 2);
        ASSERT_EQ(test_mat_copied.cols(), 4);
        ASSERT_EQ(
            test_mat_copied.get_elem(0, 0).get_scalar(),
            static_cast<TPrecision>(0.)
        );
        ASSERT_EQ(
            test_mat_copied.get_elem(0, 1).get_scalar(),
            static_cast<TPrecision>(-1.)
        );
        ASSERT_EQ(
            test_mat_copied.get_elem(0, 2).get_scalar(),
            static_cast<TPrecision>(-2.)
        );
        ASSERT_EQ(
            test_mat_copied.get_elem(0, 3).get_scalar(),
            static_cast<TPrecision>(-2.)
        );
        ASSERT_EQ(
            test_mat_copied.get_elem(1, 0).get_scalar(),
            static_cast<TPrecision>(10.)
        );
        ASSERT_EQ(
            test_mat_copied.get_elem(1, 1).get_scalar(),
            static_cast<TPrecision>(0.)
        );
        ASSERT_EQ(
            test_mat_copied.get_elem(1, 2).get_scalar(),
            static_cast<TPrecision>(12.)
        );
        ASSERT_EQ(
            test_mat_copied.get_elem(1, 3).get_scalar(),
            static_cast<TPrecision>(12.)
        );

    }

    template <typename TPrecision>
    void TestZeroMatrixCreation() {

        constexpr int m_zero(15);
        constexpr int n_zero(17);
        TMatrix<TPrecision> test_zero(TMatrix<TPrecision>::Zero(
            TestBase::bundle, m_zero, n_zero
        ));
        ASSERT_EQ(test_zero.rows(), m_zero);
        ASSERT_EQ(test_zero.cols(), n_zero);
        for (int i=0; i<m_zero; ++i) {
            for (int j=0; j<n_zero; ++j) {
                ASSERT_EQ(
                    test_zero.get_elem(i, j).get_scalar(),
                    static_cast<TPrecision>(0.)
                );
            }
        }

    }

    template <typename TPrecision>
    void TestOnesMatrixCreation() {

        constexpr int m_one(32);
        constexpr int n_one(13);
        TMatrix<TPrecision> test_ones(TMatrix<TPrecision>::Ones(
            TestBase::bundle, m_one, n_one
        ));
        ASSERT_EQ(test_ones.rows(), m_one);
        ASSERT_EQ(test_ones.cols(), n_one);
        for (int i=0; i<m_one; ++i) {
            for (int j=0; j<n_one; ++j) {
                ASSERT_EQ(
                    test_ones.get_elem(i, j).get_scalar(),
                    static_cast<TPrecision>(1.)
                );
            }
        }

    }

    template <typename TPrecision>
    void TestIdentityMatrixCreation() {

        constexpr int m_identity(40);
        constexpr int n_identity(20);
    
        TMatrix<TPrecision> test_identity_tall(TMatrix<TPrecision>::Identity(
            TestBase::bundle, m_identity, n_identity
        ));
        ASSERT_EQ(test_identity_tall.rows(), m_identity);
        ASSERT_EQ(test_identity_tall.cols(), n_identity);
        for (int i=0; i<m_identity; ++i) {
            for (int j=0; j<n_identity; ++j) {
                if (i == j) {
                    ASSERT_EQ(
                        test_identity_tall.get_elem(i, j).get_scalar(),
                        static_cast<TPrecision>(1.)
                    );
                } else {
                    ASSERT_EQ(
                        test_identity_tall.get_elem(i, j).get_scalar(),
                        static_cast<TPrecision>(0.)
                    );
                }
            }
        }

        TMatrix<TPrecision> test_identity_wide(TMatrix<TPrecision>::Identity(
            TestBase::bundle, n_identity, m_identity
        ));
        ASSERT_EQ(test_identity_wide.rows(), n_identity);
        ASSERT_EQ(test_identity_wide.cols(), m_identity);
        for (int i=0; i<n_identity; ++i) {
            for (int j=0; j<m_identity; ++j) {
                if (i == j) {
                    ASSERT_EQ(
                        test_identity_wide.get_elem(i, j).get_scalar(),
                        static_cast<TPrecision>(1.)
                    );
                } else {
                    ASSERT_EQ(
                        test_identity_wide.get_elem(i, j).get_scalar(),
                        static_cast<TPrecision>(0.)
                    );
                }
            }
        }

        TMatrix<TPrecision> test_identity_square(TMatrix<TPrecision>::Identity(
            TestBase::bundle, m_identity, m_identity
        ));
        ASSERT_EQ(test_identity_square.rows(), m_identity);
        ASSERT_EQ(test_identity_square.cols(), m_identity);
        for (int i=0; i<m_identity; ++i) {
            for (int j=0; j<m_identity; ++j) {
                if (i == j) {
                    ASSERT_EQ(
                        test_identity_square.get_elem(i, j).get_scalar(),
                        static_cast<TPrecision>(1.)
                    );
                } else {
                    ASSERT_EQ(
                        test_identity_square.get_elem(i, j).get_scalar(),
                        static_cast<TPrecision>(0.)
                    );
                }
            }
        }

    }

    template <typename TPrecision>
    void TestCol() {

        const TMatrix<TPrecision> const_mat(
            TestBase::bundle, 
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(2),
              static_cast<TPrecision>(3)},
             {static_cast<TPrecision>(4), static_cast<TPrecision>(5),
              static_cast<TPrecision>(6)},
             {static_cast<TPrecision>(7), static_cast<TPrecision>(8),
              static_cast<TPrecision>(9)},
             {static_cast<TPrecision>(10), static_cast<TPrecision>(11),
              static_cast<TPrecision>(12)}}
        );
        TMatrix<TPrecision> mat(const_mat);

        // Test Col access
        for (int j=0; j<3; ++j) {
            typename TMatrix<TPrecision>::Col col(mat.get_col(j));
            for (int i=0; i<4; ++i) {
                ASSERT_EQ(col.get_elem(i), const_mat.get_elem(i, j));
            }
        }

    }

    void TestBadCol() {

        const int m(4);
        const int n(3);
        TMatrix<double> mat(
            TestBase::bundle, 
            {{1, 2, 3},
             {4, 5, 6},
             {7, 8, 9},
             {10, 11, 12}}
        );

        // Test bad col
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() mutable { mat.get_col(-1); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() mutable { mat.get_col(n); }
        );

        // Test bad access in valid col
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() mutable { mat.get_col(0).get_elem(-1); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() mutable { mat.get_col(0).get_elem(m); }
        );

    }

    void TestBadBlock() {

        const int m(4);
        const int n(5);
        const TMatrix<double> const_mat (
            TestBase::bundle,
            {{1, 2, 3, 4, 5},
             {6, 7, 8, 9, 10},
             {11, 12, 13, 14, 15},
             {16, 17, 18, 19, 20}}
        );
        TMatrix<double> mat(const_mat);

        // Test invalid starts
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors, [=]() mutable { mat.get_block(-1, 0, 1, 1); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors, [=]() mutable { mat.get_block(m, 0, 1, 1); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors, [=]() mutable { mat.get_block(0, -1, 1, 1); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors, [=]() mutable { mat.get_block(0, n, 1, 1); }
        );

        // Test invalid sizes from 0
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors, [=]() mutable { mat.get_block(0, 0, -1, 1); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors, [=]() mutable { mat.get_block(0, 0, 1, -1); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors, [=]() mutable { mat.get_block(0, 0, m+1, 1); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors, [=]() mutable { mat.get_block(0, 0, 1, n+1); }
        );

        // Test invalid sizes from not initial index
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors, [=]() mutable { mat.get_block(1, 2, -1, 1); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors, [=]() mutable { mat.get_block(1, 2, 1, -1); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors, [=]() mutable { mat.get_block(1, 2, m, 1); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors, [=]() mutable { mat.get_block(1, 2, 1, n-1); }
        );

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

    template <typename TPrecision>
    void TestScale() {

        TMatrix<TPrecision> mat(
            TestBase::bundle,
            {{static_cast<TPrecision>(-8), static_cast<TPrecision>(0.8),
              static_cast<TPrecision>(-0.6)},
             {static_cast<TPrecision>(1), static_cast<TPrecision>(2),
              static_cast<TPrecision>(3)},
             {static_cast<TPrecision>(-3), static_cast<TPrecision>(-2),
              static_cast<TPrecision>(-1)},
             {static_cast<TPrecision>(10), static_cast<TPrecision>(100),
              static_cast<TPrecision>(1000)}}
        );
        TMatrix<TPrecision> mat_scaled_mult(
            mat*Scalar<TPrecision>(static_cast<TPrecision>(4))
        );
        for (int i=0; i<4; ++i) {
            for (int j=0; j<3; ++j) {
                ASSERT_EQ(
                    mat_scaled_mult.get_elem(i, j).get_scalar(),
                    static_cast<TPrecision>(4)*mat.get_elem(i, j).get_scalar()
                );
            }
        }
        TMatrix<TPrecision> mat_scaled_div(
            mat/Scalar<TPrecision>(static_cast<TPrecision>(10))
        );
        for (int i=0; i<4; ++i) {
            for (int j=0; j<3; ++j) {
                ASSERT_EQ(
                    mat_scaled_div.get_elem(i, j).get_scalar(),
                    ((static_cast<TPrecision>(1)/static_cast<TPrecision>(10)) *
                     mat.get_elem(i, j).get_scalar())
                );
            }
        }

    }

    template <typename TPrecision>
    void TestScaleAssignment() {

        TMatrix<TPrecision> mat(
            TestBase::bundle,
            {{static_cast<TPrecision>(-8), static_cast<TPrecision>(0.8),
              static_cast<TPrecision>(-0.6)},
             {static_cast<TPrecision>(1), static_cast<TPrecision>(2),
              static_cast<TPrecision>(3)},
             {static_cast<TPrecision>(-3), static_cast<TPrecision>(-2),
              static_cast<TPrecision>(-1)},
             {static_cast<TPrecision>(10), static_cast<TPrecision>(100),
              static_cast<TPrecision>(1000)}}
        );
        TMatrix<TPrecision> temp_mat_1(mat);
        TMatrix<TPrecision> temp_mat_2(mat);
        temp_mat_1 *= Scalar<TPrecision>(static_cast<TPrecision>(4));
        for (int i=0; i<4; ++i) {
            for (int j=0; j<3; ++j) {
                ASSERT_EQ(
                    temp_mat_1.get_elem(i, j).get_scalar(),
                    static_cast<TPrecision>(4)*mat.get_elem(i, j).get_scalar()
                );
            }
        }
        temp_mat_2 /= Scalar<TPrecision>(static_cast<TPrecision>(10));
        for (int i=0; i<4; ++i) {
            for (int j=0; j<3; ++j) {
                ASSERT_EQ(
                    temp_mat_2.get_elem(i, j).get_scalar(),
                    ((static_cast<TPrecision>(1)/static_cast<TPrecision>(10)) *
                     mat.get_elem(i, j).get_scalar())
                );
            }
        }

    }

    template <typename TPrecision>
    void TestMaxMagElem() {

        TMatrix<TPrecision> mat_all_diff(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(2),
              static_cast<TPrecision>(3)},
             {static_cast<TPrecision>(4), static_cast<TPrecision>(5),
              static_cast<TPrecision>(6)},
             {static_cast<TPrecision>(7), static_cast<TPrecision>(8),
              static_cast<TPrecision>(9)},
             {static_cast<TPrecision>(3), static_cast<TPrecision>(2),
              static_cast<TPrecision>(1)}}
        );
        ASSERT_EQ(
            mat_all_diff.get_max_mag_elem().get_scalar(),
            static_cast<TPrecision>(9)
        );

        TMatrix<TPrecision> mat_all_same(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(1),
              static_cast<TPrecision>(1)},
             {static_cast<TPrecision>(1), static_cast<TPrecision>(1),
              static_cast<TPrecision>(1)}}
        );
        ASSERT_EQ(
            mat_all_same.get_max_mag_elem().get_scalar(),
            static_cast<TPrecision>(1)
        );

        TMatrix<TPrecision> mat_pos_neg(
            TestBase::bundle,
            {{static_cast<TPrecision>(-1), static_cast<TPrecision>(-9),
              static_cast<TPrecision>(8)},
             {static_cast<TPrecision>(1), static_cast<TPrecision>(-14),
              static_cast<TPrecision>(10)}}
        );
        ASSERT_EQ(
            mat_pos_neg.get_max_mag_elem().get_scalar(),
            static_cast<TPrecision>(14)
        );

    }

    template <typename TPrecision>
    void TestNormalizeMagnitude() {

        TMatrix<TPrecision> mat_has_zeros(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(2),
              static_cast<TPrecision>(3)},
             {static_cast<TPrecision>(4), static_cast<TPrecision>(0),
              static_cast<TPrecision>(6)},
             {static_cast<TPrecision>(7), static_cast<TPrecision>(8),
              static_cast<TPrecision>(9)},
             {static_cast<TPrecision>(3), static_cast<TPrecision>(0),
              static_cast<TPrecision>(1)}}
        );
        TMatrix<TPrecision> temp_mat_has_zeros(mat_has_zeros);
        temp_mat_has_zeros.normalize_magnitude();
        ASSERT_MATRIX_NEAR(
            temp_mat_has_zeros,
            mat_has_zeros/Scalar<TPrecision>(static_cast<TPrecision>(9)),
            Tol<TPrecision>::roundoff_T()
        );

        TMatrix<TPrecision> mat_all_same(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(1),
              static_cast<TPrecision>(1)},
             {static_cast<TPrecision>(1), static_cast<TPrecision>(1),
              static_cast<TPrecision>(1)}}
        );
        TMatrix<TPrecision> temp_mat_all_same(mat_all_same);
        temp_mat_all_same.normalize_magnitude();
        ASSERT_MATRIX_NEAR(
            temp_mat_all_same,
            mat_all_same,
            Tol<TPrecision>::roundoff_T()
        );

        TMatrix<TPrecision> mat_pos_neg(
            TestBase::bundle,
            {{static_cast<TPrecision>(-1), static_cast<TPrecision>(-9),
              static_cast<TPrecision>(8)},
             {static_cast<TPrecision>(1), static_cast<TPrecision>(-14),
              static_cast<TPrecision>(10)}}
        );
        TMatrix<TPrecision> temp_mat_pos_neg(mat_pos_neg);
        temp_mat_pos_neg.normalize_magnitude();
        ASSERT_MATRIX_NEAR(
            temp_mat_pos_neg,
            mat_pos_neg/Scalar<TPrecision>(static_cast<TPrecision>(14)),
            Tol<TPrecision>::roundoff_T()
        );

    }

    template <typename TPrecision>
    void TestMatVec() {

        // Test manually
        TMatrix<TPrecision> mat(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(2),
              static_cast<TPrecision>(3)},
             {static_cast<TPrecision>(4), static_cast<TPrecision>(5),
              static_cast<TPrecision>(6)},
             {static_cast<TPrecision>(7), static_cast<TPrecision>(8),
              static_cast<TPrecision>(9)},
             {static_cast<TPrecision>(3), static_cast<TPrecision>(2),
              static_cast<TPrecision>(1)}}
        );
        ASSERT_VECTOR_NEAR(
            mat*Vector<TPrecision>(
                TestBase::bundle,
                {static_cast<TPrecision>(1), static_cast<TPrecision>(0),
                 static_cast<TPrecision>(0)}
            ),
            Vector<TPrecision>(
                TestBase::bundle,
                {static_cast<TPrecision>(1), static_cast<TPrecision>(4),
                 static_cast<TPrecision>(7), static_cast<TPrecision>(3)}
            ),
            (static_cast<TPrecision>(2.) *
             static_cast<TPrecision>(Tol<TPrecision>::gamma(3)))
        );
        ASSERT_VECTOR_NEAR(
            mat*Vector<TPrecision>(
                TestBase::bundle,
                {static_cast<TPrecision>(0), static_cast<TPrecision>(1),
                 static_cast<TPrecision>(0)}
            ),
            Vector<TPrecision>(
                TestBase::bundle,
                {static_cast<TPrecision>(2), static_cast<TPrecision>(5),
                 static_cast<TPrecision>(8), static_cast<TPrecision>(2)}
            ),
            (static_cast<TPrecision>(2.) *
             static_cast<TPrecision>(Tol<TPrecision>::gamma(3)))
        );
        ASSERT_VECTOR_NEAR(
            mat*Vector<TPrecision>(
                TestBase::bundle,
                {static_cast<TPrecision>(0), static_cast<TPrecision>(0),
                 static_cast<TPrecision>(1)}
            ),
            Vector<TPrecision>(
                TestBase::bundle,
                {static_cast<TPrecision>(3), static_cast<TPrecision>(6),
                 static_cast<TPrecision>(9), static_cast<TPrecision>(1)}
            ),
            (static_cast<TPrecision>(2.) *
             static_cast<TPrecision>(Tol<TPrecision>::gamma(3)))
        );
        ASSERT_VECTOR_NEAR(
            mat*Vector<TPrecision>(
                TestBase::bundle,
                {static_cast<TPrecision>(1), static_cast<TPrecision>(0.1),
                 static_cast<TPrecision>(0.01)}
            ),
            Vector<TPrecision>(
                TestBase::bundle,
                {static_cast<TPrecision>(1.23), static_cast<TPrecision>(4.56),
                 static_cast<TPrecision>(7.89), static_cast<TPrecision>(3.21)}
            ),
            (static_cast<TPrecision>(2.) *
             static_cast<TPrecision>(Tol<TPrecision>::gamma(3)))
        );

    }

    template <typename TPrecision>
    void TestBadMatVec() {

        TMatrix<TPrecision> mat(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(2),
              static_cast<TPrecision>(3), static_cast<TPrecision>(4)},
             {static_cast<TPrecision>(5), static_cast<TPrecision>(6),
              static_cast<TPrecision>(7), static_cast<TPrecision>(8)},
             {static_cast<TPrecision>(9), static_cast<TPrecision>(10),
              static_cast<TPrecision>(11), static_cast<TPrecision>(12)}}
        );

        Vector<TPrecision> vec_too_small(
            TestBase::bundle,
            {static_cast<TPrecision>(1), static_cast<TPrecision>(1)}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() { mat*vec_too_small; }
        );

        Vector<TPrecision> vec_too_large(
            TestBase::bundle, 
            {static_cast<TPrecision>(1), static_cast<TPrecision>(1),
             static_cast<TPrecision>(1), static_cast<TPrecision>(1),
             static_cast<TPrecision>(1)}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() { mat*vec_too_large; }
        );

        Vector<TPrecision> vec_matches_wrong_dimension(
            TestBase::bundle,
            {static_cast<TPrecision>(1), static_cast<TPrecision>(1),
             static_cast<TPrecision>(1)}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() { mat*vec_matches_wrong_dimension; }
        );

    }

    template <typename TPrecision>
    void TestTransposeMatVec() {

        // Test manually
        TMatrix<TPrecision> mat(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(2),
              static_cast<TPrecision>(3), static_cast<TPrecision>(4)},
             {static_cast<TPrecision>(5), static_cast<TPrecision>(6),
              static_cast<TPrecision>(7), static_cast<TPrecision>(8)},
             {static_cast<TPrecision>(9), static_cast<TPrecision>(1),
              static_cast<TPrecision>(2), static_cast<TPrecision>(3)}}
        );
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod(
                Vector<TPrecision>(
                    TestBase::bundle,
                    {static_cast<TPrecision>(1), static_cast<TPrecision>(0),
                     static_cast<TPrecision>(0)}
                )
            ),
            Vector<TPrecision>(
                TestBase::bundle,
                {static_cast<TPrecision>(1), static_cast<TPrecision>(2),
                 static_cast<TPrecision>(3), static_cast<TPrecision>(4)}
            ),
            (static_cast<TPrecision>(2.) *
             static_cast<TPrecision>(Tol<TPrecision>::gamma(3)))
        );
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod(
                Vector<TPrecision>(
                    TestBase::bundle,
                    {static_cast<TPrecision>(0), static_cast<TPrecision>(1),
                     static_cast<TPrecision>(0)}
                )
            ),
            Vector<TPrecision>(
                TestBase::bundle,
                {static_cast<TPrecision>(5), static_cast<TPrecision>(6),
                 static_cast<TPrecision>(7), static_cast<TPrecision>(8)}
            ),
            (static_cast<TPrecision>(2.) *
             static_cast<TPrecision>(Tol<TPrecision>::gamma(3)))
        );
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod(
                Vector<TPrecision>(
                    TestBase::bundle,
                    {static_cast<TPrecision>(0), static_cast<TPrecision>(0),
                     static_cast<TPrecision>(1)}
                )
            ),
            Vector<TPrecision>(
                TestBase::bundle,
                {static_cast<TPrecision>(9), static_cast<TPrecision>(1),
                 static_cast<TPrecision>(2), static_cast<TPrecision>(3)}
            ),
            (static_cast<TPrecision>(2.) *
             static_cast<TPrecision>(Tol<TPrecision>::gamma(3)))
        );
        ASSERT_VECTOR_NEAR(
            mat.transpose_prod(
                Vector<TPrecision>(
                    TestBase::bundle,
                    {static_cast<TPrecision>(1), static_cast<TPrecision>(0.1),
                     static_cast<TPrecision>(0.01)}
                )
            ),
            Vector<TPrecision>(
                TestBase::bundle,
                {static_cast<TPrecision>(1.59), static_cast<TPrecision>(2.61),
                 static_cast<TPrecision>(3.72), static_cast<TPrecision>(4.83)}
            ),
            (static_cast<TPrecision>(2.) *
             static_cast<TPrecision>(Tol<TPrecision>::gamma(3)))
        );

    }

    template <typename TPrecision>
    void TestBadTransposeMatVec() {

        TMatrix<TPrecision> mat(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(2),
              static_cast<TPrecision>(3), static_cast<TPrecision>(4)},
             {static_cast<TPrecision>(5), static_cast<TPrecision>(6),
              static_cast<TPrecision>(7), static_cast<TPrecision>(8)},
             {static_cast<TPrecision>(9), static_cast<TPrecision>(10),
              static_cast<TPrecision>(11), static_cast<TPrecision>(12)}}
        );

        Vector<TPrecision> vec_too_small(
            TestBase::bundle,
            {static_cast<TPrecision>(1), static_cast<TPrecision>(1)}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() { mat.transpose_prod(vec_too_small); }
        );

        Vector<TPrecision> vec_too_large(
            TestBase::bundle,
            {static_cast<TPrecision>(1), static_cast<TPrecision>(1),
             static_cast<TPrecision>(1), static_cast<TPrecision>(1),
             static_cast<TPrecision>(1)}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() { mat.transpose_prod(vec_too_large); }
        );

        Vector<TPrecision> vec_matches_wrong_dimension(
            TestBase::bundle,
            {static_cast<TPrecision>(1), static_cast<TPrecision>(1),
             static_cast<TPrecision>(1), static_cast<TPrecision>(1)}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() { mat.transpose_prod(vec_matches_wrong_dimension); }
        );

    }

    template <typename TPrecision>
    void TestTranspose() {

        // Test manually
        constexpr int m_manual(4);
        constexpr int n_manual(3);
        TMatrix<TPrecision> mat(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(2),
              static_cast<TPrecision>(3), static_cast<TPrecision>(4)},
             {static_cast<TPrecision>(5), static_cast<TPrecision>(6),
              static_cast<TPrecision>(7), static_cast<TPrecision>(8)},
             {static_cast<TPrecision>(9), static_cast<TPrecision>(10),
              static_cast<TPrecision>(11), static_cast<TPrecision>(12)}}
        );

        TMatrix<TPrecision> mat_transposed(mat.transpose());
        ASSERT_EQ(mat_transposed.rows(), m_manual);
        ASSERT_EQ(mat_transposed.cols(), n_manual);
        for (int i=0; i<m_manual; ++i) {
            for (int j=0; j<n_manual; ++j) {
                ASSERT_EQ(
                    mat_transposed.get_elem(i, j).get_scalar(),
                    mat.get_elem(j, i).get_scalar()
                );
            }
        }

        TMatrix<TPrecision> test(
            TestBase::bundle, 
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(5),
              static_cast<TPrecision>(9)},
             {static_cast<TPrecision>(2), static_cast<TPrecision>(6),
              static_cast<TPrecision>(10)},
             {static_cast<TPrecision>(3), static_cast<TPrecision>(7),
              static_cast<TPrecision>(11)},
             {static_cast<TPrecision>(4), static_cast<TPrecision>(8),
              static_cast<TPrecision>(12)}}
        );
        ASSERT_MATRIX_EQ(mat_transposed, test);

        // Test sparse manually
        constexpr int m_sparse_manual(3);
        constexpr int n_sparse_manual(4);
        TMatrix<TPrecision> mat_sparse(
            TestBase::bundle,
            {{static_cast<TPrecision>(6), static_cast<TPrecision>(0),
              static_cast<TPrecision>(0)},
             {static_cast<TPrecision>(3), static_cast<TPrecision>(2),
              static_cast<TPrecision>(1)},
             {static_cast<TPrecision>(0), static_cast<TPrecision>(-1),
              static_cast<TPrecision>(0)},
             {static_cast<TPrecision>(-3), static_cast<TPrecision>(-4),
              static_cast<TPrecision>(0)}}
        );

        TMatrix<TPrecision> mat_sparse_transposed(mat_sparse.transpose());
        ASSERT_EQ(mat_sparse_transposed.rows(), m_sparse_manual);
        ASSERT_EQ(mat_sparse_transposed.cols(), n_sparse_manual);
        for (int i=0; i<m_sparse_manual; ++i) {
            for (int j=0; j<n_sparse_manual; ++j) {
                ASSERT_EQ(
                    mat_sparse_transposed.get_elem(i, j).get_scalar(),
                    mat_sparse.get_elem(j, i).get_scalar()
                );
            }
        }

        TMatrix<TPrecision> test_sparse(
            TestBase::bundle,
            {{static_cast<TPrecision>(6), static_cast<TPrecision>(3),
              static_cast<TPrecision>(0), static_cast<TPrecision>(-3)},
             {static_cast<TPrecision>(0), static_cast<TPrecision>(2),
              static_cast<TPrecision>(-1), static_cast<TPrecision>(-4)},
             {static_cast<TPrecision>(0), static_cast<TPrecision>(1),
              static_cast<TPrecision>(0), static_cast<TPrecision>(0)}}
        );
        ASSERT_MATRIX_EQ(mat_sparse_transposed, test_sparse);

    }

    template <typename TPrecision>
    void TestMatMat() {

        TMatrix<TPrecision> mat1(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(4)},
             {static_cast<TPrecision>(-1), static_cast<TPrecision>(-4)},
             {static_cast<TPrecision>(-3), static_cast<TPrecision>(-2)},
             {static_cast<TPrecision>(10), static_cast<TPrecision>(100)}}
        );
        TMatrix<TPrecision> mat2(
            TestBase::bundle,
            {{static_cast<TPrecision>(7), static_cast<TPrecision>(2),
              static_cast<TPrecision>(-3)},
             {static_cast<TPrecision>(10), static_cast<TPrecision>(-4),
              static_cast<TPrecision>(-3)}}
        );
        TMatrix<TPrecision> test_mat(
            TestBase::bundle,
            {{static_cast<TPrecision>(47), static_cast<TPrecision>(-14),
              static_cast<TPrecision>(-15)},
             {static_cast<TPrecision>(-47), static_cast<TPrecision>(14),
              static_cast<TPrecision>(15)},
             {static_cast<TPrecision>(-41), static_cast<TPrecision>(2),
              static_cast<TPrecision>(15)},
             {static_cast<TPrecision>(1070), static_cast<TPrecision>(-380),
              static_cast<TPrecision>(-330)}}
        );
        ASSERT_MATRIX_EQ(mat1*mat2, test_mat);
        ASSERT_MATRIX_EQ(
            mat1*TMatrix<TPrecision>::Identity(TestBase::bundle, 2, 2),
            mat1
        );
        ASSERT_MATRIX_EQ(
            mat2*TMatrix<TPrecision>::Identity(TestBase::bundle, 3, 3),
            mat2
        );

    }

    template <typename TPrecision>
    void TestBadMatMat() {

        TMatrix<TPrecision> mat(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(4)},
             {static_cast<TPrecision>(-1), static_cast<TPrecision>(-4)},
             {static_cast<TPrecision>(-3), static_cast<TPrecision>(-2)},
             {static_cast<TPrecision>(10), static_cast<TPrecision>(100)}}
        );

        TMatrix<TPrecision> mat_too_small(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(1)}}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors, [=]() { mat*mat_too_small; }
        );

        TMatrix<TPrecision> mat_too_big(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(1)},
             {static_cast<TPrecision>(1), static_cast<TPrecision>(1)},
             {static_cast<TPrecision>(1), static_cast<TPrecision>(1)}}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors, [=]() { mat*mat_too_big; }
        );

        TMatrix<TPrecision> mat_matches_other_dim(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(1),
              static_cast<TPrecision>(1), static_cast<TPrecision>(1)},
             {static_cast<TPrecision>(1), static_cast<TPrecision>(1),
              static_cast<TPrecision>(1), static_cast<TPrecision>(1)},
             {static_cast<TPrecision>(1), static_cast<TPrecision>(1),
              static_cast<TPrecision>(1), static_cast<TPrecision>(1)},
             {static_cast<TPrecision>(1), static_cast<TPrecision>(1),
              static_cast<TPrecision>(1), static_cast<TPrecision>(1)}}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors, [=]() { mat*mat_matches_other_dim; }
        );

    }

    template <typename TPrecision>
    void TestAddSub() {

        TMatrix<TPrecision> mat1(
            TestBase::bundle,
            {{static_cast<TPrecision>(47), static_cast<TPrecision>(-14),
              static_cast<TPrecision>(-15)},
             {static_cast<TPrecision>(-47), static_cast<TPrecision>(14),
              static_cast<TPrecision>(15)},
             {static_cast<TPrecision>(-41), static_cast<TPrecision>(2),
              static_cast<TPrecision>(15)},
             {static_cast<TPrecision>(10), static_cast<TPrecision>(-38),
              static_cast<TPrecision>(-33)}}
        );

        TMatrix<TPrecision> mat2(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(2),
              static_cast<TPrecision>(3)},
             {static_cast<TPrecision>(-4), static_cast<TPrecision>(-5),
              static_cast<TPrecision>(-6)},
             {static_cast<TPrecision>(-7), static_cast<TPrecision>(8),
              static_cast<TPrecision>(9)},
             {static_cast<TPrecision>(10), static_cast<TPrecision>(-11),
              static_cast<TPrecision>(-12)}}
        );

        TMatrix<TPrecision> test_mat_add(
            TestBase::bundle,
            {{static_cast<TPrecision>(48), static_cast<TPrecision>(-12),
              static_cast<TPrecision>(-12)},
             {static_cast<TPrecision>(-51), static_cast<TPrecision>(9),
              static_cast<TPrecision>(9)},
             {static_cast<TPrecision>(-48), static_cast<TPrecision>(10),
              static_cast<TPrecision>(24)},
             {static_cast<TPrecision>(20), static_cast<TPrecision>(-49),
              static_cast<TPrecision>(-45)}}
        );
        ASSERT_MATRIX_EQ(mat1+mat2, test_mat_add);

        TMatrix<TPrecision> test_mat_sub_1(
            TestBase::bundle,
            {{static_cast<TPrecision>(46), static_cast<TPrecision>(-16),
              static_cast<TPrecision>(-18)},
             {static_cast<TPrecision>(-43), static_cast<TPrecision>(19),
              static_cast<TPrecision>(21)},
             {static_cast<TPrecision>(-34), static_cast<TPrecision>(-6),
              static_cast<TPrecision>(6)},
             {static_cast<TPrecision>(0), static_cast<TPrecision>(-27),
              static_cast<TPrecision>(-21)}}
        );
        ASSERT_MATRIX_EQ(mat1-mat2, test_mat_sub_1);

        TMatrix<TPrecision> test_mat_sub_2(
            TestBase::bundle,
            {{static_cast<TPrecision>(-46), static_cast<TPrecision>(16),
              static_cast<TPrecision>(18)},
             {static_cast<TPrecision>(43), static_cast<TPrecision>(-19),
              static_cast<TPrecision>(-21)},
             {static_cast<TPrecision>(34), static_cast<TPrecision>(6),
              static_cast<TPrecision>(-6)},
             {static_cast<TPrecision>(0), static_cast<TPrecision>(27),
              static_cast<TPrecision>(21)}}
        );
        ASSERT_MATRIX_EQ(mat2-mat1, test_mat_sub_2);

        ASSERT_MATRIX_EQ(
            mat1-mat1,
            TMatrix<TPrecision>::Zero(TestBase::bundle, 4, 3)
        );
        ASSERT_MATRIX_EQ(
            mat2-mat2,
            TMatrix<TPrecision>::Zero(TestBase::bundle, 4, 3)
        );

    }

    template <typename TPrecision>
    void TestBadAddSub() {

        TMatrix<TPrecision> mat(
            TestBase::bundle,
            {{static_cast<TPrecision>(47), static_cast<TPrecision>(-14),
              static_cast<TPrecision>(-15)},
             {static_cast<TPrecision>(-47), static_cast<TPrecision>(14),
              static_cast<TPrecision>(15)},
             {static_cast<TPrecision>(-41), static_cast<TPrecision>(2),
              static_cast<TPrecision>(15)},
             {static_cast<TPrecision>(10), static_cast<TPrecision>(-38),
              static_cast<TPrecision>(-33)}}
        );

        TMatrix<TPrecision> mat_too_few_cols(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(1)},
             {static_cast<TPrecision>(1), static_cast<TPrecision>(1)},
             {static_cast<TPrecision>(1), static_cast<TPrecision>(1)},
             {static_cast<TPrecision>(1), static_cast<TPrecision>(1)}}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors, [=]() { mat+mat_too_few_cols; }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors, [=]() { mat-mat_too_few_cols; }
        );

        TMatrix<TPrecision> mat_too_many_cols(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(1),
              static_cast<TPrecision>(1), static_cast<TPrecision>(1)},
             {static_cast<TPrecision>(1), static_cast<TPrecision>(1),
              static_cast<TPrecision>(1), static_cast<TPrecision>(1)},
             {static_cast<TPrecision>(1), static_cast<TPrecision>(1),
              static_cast<TPrecision>(1), static_cast<TPrecision>(1)},
             {static_cast<TPrecision>(1), static_cast<TPrecision>(1),
              static_cast<TPrecision>(1), static_cast<TPrecision>(1)}}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors, [=]() { mat+mat_too_many_cols; }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors, [=]() { mat-mat_too_many_cols; }
        );

        TMatrix<TPrecision> mat_too_few_rows(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(1),
              static_cast<TPrecision>(1)},
             {static_cast<TPrecision>(1), static_cast<TPrecision>(1),
              static_cast<TPrecision>(1)},
             {static_cast<TPrecision>(1), static_cast<TPrecision>(1),
              static_cast<TPrecision>(1)}}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors, [=]() { mat+mat_too_few_rows; }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors, [=]() { mat-mat_too_few_rows; }
        );

        TMatrix<TPrecision> mat_too_many_rows(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(1),
              static_cast<TPrecision>(1)},
             {static_cast<TPrecision>(1), static_cast<TPrecision>(1),
              static_cast<TPrecision>(1)},
             {static_cast<TPrecision>(1), static_cast<TPrecision>(1),
              static_cast<TPrecision>(1)},
             {static_cast<TPrecision>(1), static_cast<TPrecision>(1),
              static_cast<TPrecision>(1)},
             {static_cast<TPrecision>(1), static_cast<TPrecision>(1),
              static_cast<TPrecision>(1)}}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors, [=]() { mat+mat_too_many_rows; }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors, [=]() { mat-mat_too_many_rows; }
        );

    }

    template <typename TPrecision>
    void TestNorm() {

        TMatrix<TPrecision> mat1(
            TestBase::bundle,
            {{static_cast<TPrecision>(18), static_cast<TPrecision>(5)},
             {static_cast<TPrecision>(-3), static_cast<TPrecision>(6)},
             {static_cast<TPrecision>(9), static_cast<TPrecision>(-9)},
             {static_cast<TPrecision>(4), static_cast<TPrecision>(-2)}}
        );
        ASSERT_EQ(mat1.norm().get_scalar(), static_cast<TPrecision>(24));

        TMatrix<TPrecision> mat2(
            TestBase::bundle,
            {{static_cast<TPrecision>(1), static_cast<TPrecision>(-1),
              static_cast<TPrecision>(1)},
             {static_cast<TPrecision>(1), static_cast<TPrecision>(1),
              static_cast<TPrecision>(-1)},
             {static_cast<TPrecision>(-1), static_cast<TPrecision>(-1),
              static_cast<TPrecision>(-1)}}
        );
        ASSERT_EQ(mat2.norm().get_scalar(), static_cast<TPrecision>(3));

    }

    void TestCast() {

        const int m = 15;
        const int n = 12;

        TMatrix<double> mat_dbl(
            CommonMatRandomInterface<TMatrix, double>::rand_matrix(
                TestBase::bundle, m, n
            )
        );

        TMatrix<double> dbl_to_dbl(mat_dbl.template cast<double>());
        ASSERT_MATRIX_EQ(dbl_to_dbl, mat_dbl);

        TMatrix<float> dbl_to_sgl(mat_dbl.template cast<float>());
        ASSERT_EQ(dbl_to_sgl.rows(), m);
        ASSERT_EQ(dbl_to_sgl.cols(), n);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_NEAR(
                    dbl_to_sgl.get_elem(i, j).get_scalar(),
                    static_cast<float>(mat_dbl.get_elem(i, j).get_scalar()),
                    (min_1_mag(static_cast<float>(
                     mat_dbl.get_elem(i, j).get_scalar())) *
                     Tol<float>::roundoff_T())
                );
            }
        }

        TMatrix<__half> dbl_to_hlf(mat_dbl.template cast<__half>());
        ASSERT_EQ(dbl_to_hlf.rows(), m);
        ASSERT_EQ(dbl_to_hlf.cols(), n);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_NEAR(
                    dbl_to_hlf.get_elem(i, j).get_scalar(),
                    static_cast<__half>(mat_dbl.get_elem(i, j).get_scalar()),
                    (min_1_mag(static_cast<__half>(
                     mat_dbl.get_elem(i, j).get_scalar())) *
                     Tol<__half>::roundoff_T())
                );
            }
        }

        TMatrix<float> mat_sgl(
            CommonMatRandomInterface<TMatrix, float>::rand_matrix(
                TestBase::bundle, m, n
            )
        );

        TMatrix<float> sgl_to_sgl(mat_sgl.template cast<float>());
        ASSERT_MATRIX_EQ(sgl_to_sgl, mat_sgl);
    
        TMatrix<double> sgl_to_dbl(mat_sgl.template cast<double>());
        ASSERT_EQ(sgl_to_dbl.rows(), m);
        ASSERT_EQ(sgl_to_dbl.cols(), n);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_NEAR(
                    sgl_to_dbl.get_elem(i, j).get_scalar(),
                    static_cast<double>(mat_sgl.get_elem(i, j).get_scalar()),
                    (min_1_mag(static_cast<double>(
                     mat_sgl.get_elem(i, j).get_scalar())) *
                     static_cast<double>(Tol<float>::roundoff_T()))
                );
            }
        }

        TMatrix<__half> sgl_to_hlf(mat_sgl.template cast<__half>());
        ASSERT_EQ(sgl_to_hlf.rows(), m);
        ASSERT_EQ(sgl_to_hlf.cols(), n);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_NEAR(
                    sgl_to_hlf.get_elem(i, j).get_scalar(),
                    static_cast<__half>(mat_sgl.get_elem(i, j).get_scalar()),
                    (min_1_mag(static_cast<__half>(
                     mat_sgl.get_elem(i, j).get_scalar())) *
                     Tol<__half>::roundoff_T())
                );
            }
        }

        TMatrix<__half> mat_hlf(
            CommonMatRandomInterface<TMatrix, __half>::rand_matrix(
                TestBase::bundle, m, n
            )
        );

        TMatrix<__half> hlf_to_hlf(mat_hlf.template cast<__half>());
        ASSERT_MATRIX_EQ(hlf_to_hlf, mat_hlf);

        TMatrix<float> hlf_to_sgl(mat_hlf.template cast<float>());
        ASSERT_EQ(hlf_to_sgl.rows(), m);
        ASSERT_EQ(hlf_to_sgl.cols(), n);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_NEAR(
                    hlf_to_sgl.get_elem(i, j).get_scalar(),
                    static_cast<float>(mat_hlf.get_elem(i, j).get_scalar()),
                    (min_1_mag(static_cast<float>(
                     mat_hlf.get_elem(i, j).get_scalar())) *
                     static_cast<float>(Tol<__half>::roundoff_T()))
                );
            }
        }

        TMatrix<double> hlf_to_dbl(mat_hlf.template cast<double>());
        ASSERT_EQ(hlf_to_dbl.rows(), m);
        ASSERT_EQ(hlf_to_dbl.cols(), n);
        for (int i=0; i<m; ++i) {
            for (int j=0; j<n; ++j) {
                ASSERT_NEAR(
                    hlf_to_dbl.get_elem(i, j).get_scalar(),
                    static_cast<double>(mat_hlf.get_elem(i, j).get_scalar()),
                    (min_1_mag(static_cast<double>(
                     mat_hlf.get_elem(i, j).get_scalar())) *
                     static_cast<double>(Tol<__half>::roundoff_T()))
                );
            }
        }

    }

};

template <template <typename> typename TMatrix>
class Matrix_Substitution_Test: public TestBase
{
public:

    template <typename TPrecision>
    void TestForwardSubstitution() {

        const int n(90);

        TMatrix<TPrecision> L_tri(read_matrixCSV<TMatrix, TPrecision>(
            TestBase::bundle, solve_matrix_dir / fs::path("L_tri_90.csv")
        ));
        Vector<TPrecision> x_tri(read_vectorCSV<TPrecision>(
            TestBase::bundle, solve_matrix_dir / fs::path("x_tri_90.csv")
        ));
        Vector<TPrecision> Lb_tri(read_vectorCSV<TPrecision>(
            TestBase::bundle, solve_matrix_dir / fs::path("Lb_tri_90.csv")
        ));

        Vector<TPrecision> test_soln(L_tri.frwd_sub(Lb_tri));

        ASSERT_VECTOR_NEAR(
            test_soln,
            x_tri,
            x_tri.get_max_mag_elem().get_scalar()*Tol<TPrecision>::gamma_T(n)
        );

    }
    
    template <typename TPrecision>
    void TestRandomForwardSubstitution() {

        srand(time(NULL));
        const int n(100 + (rand() % 100));

        TMatrix<TPrecision> L_tri(
            MatrixDense<TPrecision>::Random_LT(TestBase::bundle, n, n)
        );

        Vector<TPrecision> x_tri(
            Vector<TPrecision>::Random(TestBase::bundle, n)
        );
        Vector<TPrecision> Lb_tri(L_tri*x_tri);

        Vector<TPrecision> test_soln(L_tri.frwd_sub(Lb_tri));

        ASSERT_VECTOR_NEAR(
            test_soln,
            x_tri,
            x_tri.get_max_mag_elem().get_scalar()*Tol<TPrecision>::gamma_T(n)
        );

    }

    template <typename TPrecision>
    void TestRandomSparseForwardSubstitution() {

        srand(time(NULL));
        const int n((rand() % 100)+100);

        TMatrix<TPrecision> L_tri(
            NoFillMatrixSparse<double>::Random_LT(
                TestBase::bundle,
                n, n,
                sqrt(static_cast<double>(n))/static_cast<double>(n)
            ).template cast<TPrecision>()
        );
        Vector<TPrecision> x_tri(
            Vector<TPrecision>::Random(TestBase::bundle, n)
        );
        Vector<TPrecision> Lb_tri(L_tri*x_tri);

        Vector<TPrecision> test_soln(L_tri.frwd_sub(Lb_tri));

        ASSERT_VECTOR_NEAR(
            test_soln,
            x_tri,
            x_tri.get_max_mag_elem().get_scalar()*Tol<TPrecision>::gamma_T(n)
        );

    }

    template <typename TPrecision>
    void TestMultiBlockRandomForwardSubstitution() {

        srand(time(NULL));
        const int n(
            genmat_gpu_const::MAXTHREADSPERBLOCK + (rand() % 100) + 100
        );

        TMatrix<TPrecision> L_tri(
            NoFillMatrixSparse<double>::Random_LT(
                TestBase::bundle,
                n, n,
                sqrt(static_cast<double>(n))/static_cast<double>(n)
            ).template cast<TPrecision>()
        );
        Vector<TPrecision> x_tri(
            Vector<TPrecision>::Random(TestBase::bundle, n)
        );
        Vector<TPrecision> Lb_tri(L_tri*x_tri);

        Vector<TPrecision> test_soln(L_tri.frwd_sub(Lb_tri));

        TPrecision min_tol = Tol<TPrecision>::gamma_T(n);
        if (abs_ns::abs(min_tol) > static_cast<TPrecision>(0.1)) {
            min_tol = 0.2;
        }

        ASSERT_VECTOR_NEAR(
            test_soln,
            x_tri,
            x_tri.get_max_mag_elem().get_scalar()*min_tol
        );

    }

    template <typename TPrecision>
    void TestBackwardSubstitution() {

        constexpr int n(90);

        TMatrix<TPrecision> U_tri(read_matrixCSV<TMatrix, TPrecision>(
                TestBase::bundle, solve_matrix_dir / fs::path("U_tri_90.csv")
        ));
        Vector<TPrecision> x_tri(read_vectorCSV<TPrecision>(
            TestBase::bundle, solve_matrix_dir / fs::path("x_tri_90.csv")
        ));
        Vector<TPrecision> Ub_tri(read_vectorCSV<TPrecision>(
            TestBase::bundle, solve_matrix_dir / fs::path("Ub_tri_90.csv")
        ));
    
        Vector<TPrecision> test_soln(U_tri.back_sub(Ub_tri));

        ASSERT_VECTOR_NEAR(
            test_soln,
            x_tri,
            x_tri.get_max_mag_elem().get_scalar()*Tol<TPrecision>::gamma_T(n)
        );

    }
    
    template <typename TPrecision>
    void TestRandomBackwardSubstitution() {

        srand(time(NULL));
        const int n((rand() % 100)+100);

        TMatrix<TPrecision> U_tri(
            MatrixDense<TPrecision>::Random_UT(TestBase::bundle, n, n)
        );
        Vector<TPrecision> x_tri(
            Vector<TPrecision>::Random(TestBase::bundle, n)
        );
        Vector<TPrecision> Ub_tri(U_tri*x_tri);
    
        Vector<TPrecision> test_soln(U_tri.back_sub(Ub_tri));

        ASSERT_VECTOR_NEAR(
            test_soln,
            x_tri,
            x_tri.get_max_mag_elem().get_scalar()*Tol<TPrecision>::gamma_T(n)
        );

    }

    template <typename TPrecision>
    void TestRandomSparseBackwardSubstitution() {

        srand(time(NULL));
        const int n((rand() % 100)+100);

        TMatrix<TPrecision> U_tri(
            NoFillMatrixSparse<double>::Random_UT(
                TestBase::bundle,
                n, n,
                sqrt(static_cast<double>(n))/static_cast<double>(n)
            ).template cast<TPrecision>()
        );
        Vector<TPrecision> x_tri(
            Vector<TPrecision>::Random(TestBase::bundle, n)
        );
        Vector<TPrecision> Ub_tri(U_tri*x_tri);

        Vector<TPrecision> test_soln(U_tri.back_sub(Ub_tri));

        ASSERT_VECTOR_NEAR(
            test_soln,
            x_tri,
            x_tri.get_max_mag_elem().get_scalar()*Tol<TPrecision>::gamma_T(n)
        );

    }

    template <typename TPrecision>
    void TestMultiBlockRandomBackwardSubstitution() {

        srand(time(NULL));
        const int n(
            genmat_gpu_const::MAXTHREADSPERBLOCK + (rand() % 100) + 100
        );

        TMatrix<TPrecision> U_tri(
            MatrixDense<TPrecision>::Random_UT(TestBase::bundle, n, n)
        );
        Vector<TPrecision> x_tri(
            Vector<TPrecision>::Random(TestBase::bundle, n)
        );
        Vector<TPrecision> Ub_tri(U_tri*x_tri);
    
        Vector<TPrecision> test_soln(U_tri.back_sub(Ub_tri));

        TPrecision min_tol = Tol<TPrecision>::gamma_T(n);
        if (abs_ns::abs(min_tol) > static_cast<TPrecision>(0.1)) {
            min_tol = 0.25;
        }

        ASSERT_VECTOR_NEAR(
            test_soln,
            x_tri,
            x_tri.get_max_mag_elem().get_scalar()*min_tol
        );

    }

};