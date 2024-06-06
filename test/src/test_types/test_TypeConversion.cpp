#include "test.h"

#include "types/types.h"

class TypeConversion_Test: public TestBase
{
public:

    template <typename T>
    void TestDenseToNoFillSparse() {

        // Test manual
        constexpr int m_manual(3);
        constexpr int n_manual(4);
        MatrixDense<T> dense_manual(
            TestBase::bundle,
            {{static_cast<T>(1), static_cast<T>(0), static_cast<T>(3), static_cast<T>(0)},
             {static_cast<T>(5), static_cast<T>(0), static_cast<T>(7), static_cast<T>(8)},
             {static_cast<T>(9), static_cast<T>(10), static_cast<T>(11), static_cast<T>(0)}}
        );

        NoFillMatrixSparse<T> sparse_manual(dense_manual);

        ASSERT_EQ(sparse_manual.rows(), m_manual);
        ASSERT_EQ(sparse_manual.cols(), n_manual);
        ASSERT_EQ(sparse_manual.non_zeros(), 8);
        for (int i=0; i<m_manual; ++i) {
            for (int j=0; j<n_manual; ++j) {
                ASSERT_EQ(sparse_manual.get_elem(i, j), dense_manual.get_elem(i, j));
            }
        }

    }

    template <typename T>
    void TestDenseToNoFillSparse_LONGRUNTIME() {

        // Test random
        srand(time(NULL));
        const int m_random((rand() % 50) + 100);
        const int n_random((rand() % 50) + 100);
        MatrixDense<T> dense_rand(MatrixDense<T>::Random(TestBase::bundle, m_random, n_random));

        NoFillMatrixSparse<T> sparse_rand(dense_rand);

        ASSERT_EQ(sparse_rand.rows(), m_random);
        ASSERT_EQ(sparse_rand.cols(), n_random);
        ASSERT_EQ(sparse_rand.non_zeros(), dense_rand.non_zeros());
        for (int i=0; i<m_random; ++i) {
            for (int j=0; j<n_random; ++j) {
                ASSERT_EQ(sparse_rand.get_elem(i, j), dense_rand.get_elem(i, j));
            }
        }

    }
    
    template <typename T>
    void TestNoFillSparseToDense() {

        std::initializer_list<std::initializer_list<T>> li = {
            {static_cast<T>(1), static_cast<T>(2), static_cast<T>(3),
             static_cast<T>(4), static_cast<T>(0)},
            {static_cast<T>(0), static_cast<T>(0), static_cast<T>(8),
             static_cast<T>(9), static_cast<T>(0)},
            {static_cast<T>(0), static_cast<T>(0), static_cast<T>(0),
             static_cast<T>(0), static_cast<T>(0)},
            {static_cast<T>(0), static_cast<T>(0), static_cast<T>(0),
             static_cast<T>(19), static_cast<T>(20)}
        };

        const NoFillMatrixSparse<T> nofillsparse_mat(TestBase::bundle, li);
        const MatrixDense<T> target_mat(TestBase::bundle, li);

        MatrixDense<T> test_mat(nofillsparse_mat);

        ASSERT_MATRIX_EQ(target_mat, test_mat);

    }

    template <typename T>
    void TestNoFillSparseBlockToDense() {

        const NoFillMatrixSparse<T> mat (
            TestBase::bundle,
            {{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3),
              static_cast<T>(4), static_cast<T>(0)},
             {static_cast<T>(0), static_cast<T>(0), static_cast<T>(8),
              static_cast<T>(9), static_cast<T>(0)},
             {static_cast<T>(0), static_cast<T>(0), static_cast<T>(0),
              static_cast<T>(0), static_cast<T>(0)},
             {static_cast<T>(0), static_cast<T>(0), static_cast<T>(0),
              static_cast<T>(19), static_cast<T>(20)}}
        );

        // Test MatrixDense cast/access for block 0, 0, 3, 4
        MatrixDense<T> mat_0_0_3_4(mat.get_block(0, 0, 3, 4));
        MatrixDense<T> test_0_0_3_4(
            TestBase::bundle,
            {{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4)},
             {static_cast<T>(0), static_cast<T>(0), static_cast<T>(8), static_cast<T>(9)},
             {static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)}}
        );
        ASSERT_MATRIX_EQ(mat_0_0_3_4, test_0_0_3_4);

        // Test MatrixDense cast/access for block 1, 2, 3, 1
        MatrixDense<T> mat_1_2_3_1(mat.get_block(1, 2, 3, 1));
        MatrixDense<T> test_1_2_3_1(
            TestBase::bundle,
            {{static_cast<T>(8)},
             {static_cast<T>(0)},
             {static_cast<T>(0)}}
        );
        ASSERT_MATRIX_EQ(mat_1_2_3_1, test_1_2_3_1);

        // Test MatrixDense cast/access for block 1, 2, 3, 3
        MatrixDense<T> mat_1_2_3_3(mat.get_block(1, 2, 3, 3));
        MatrixDense<T> test_1_2_3_3(
            TestBase::bundle,
            {{static_cast<T>(8), static_cast<T>(9), static_cast<T>(0)},
             {static_cast<T>(0), static_cast<T>(0), static_cast<T>(0)},
             {static_cast<T>(0), static_cast<T>(19), static_cast<T>(20)}}
        );
        ASSERT_MATRIX_EQ(mat_1_2_3_3, test_1_2_3_3);

        // Test MatrixDense cast/access for block 0, 0, 3, 4
        MatrixDense<T> mat_0_0_3_4_copy(mat.get_block(0, 0, 3, 4).copy_to_mat());
        ASSERT_MATRIX_EQ(mat_0_0_3_4_copy, test_0_0_3_4);

        // Test MatrixDense cast/access for block 1, 2, 3, 1
        MatrixDense<T> mat_1_2_3_1_copy(mat.get_block(1, 2, 3, 1).copy_to_mat());
        ASSERT_MATRIX_EQ(mat_1_2_3_1_copy, test_1_2_3_1);

        // Test MatrixDense cast/access for block 1, 2, 3, 3
        MatrixDense<T> mat_1_2_3_3_copy(mat.get_block(1, 2, 3, 3).copy_to_mat());
        ASSERT_MATRIX_EQ(mat_1_2_3_3_copy, test_1_2_3_3);

    }

    template <template <typename> typename M, typename T>
    void TestMatrixColToVector() {

        M<T> mat(
            TestBase::bundle,
            {{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4)},
             {static_cast<T>(5), static_cast<T>(6), static_cast<T>(7), static_cast<T>(8)},
             {static_cast<T>(9), static_cast<T>(10), static_cast<T>(11), static_cast<T>(12)}}
        );

        Vector<T> vec_col_0(mat.get_col(0).copy_to_vec());
        Vector<T> test_vec_col_0(
            TestBase::bundle,
            {static_cast<T>(1), static_cast<T>(5), static_cast<T>(9)}
        );
        ASSERT_VECTOR_EQ(vec_col_0, test_vec_col_0);

        Vector<T> vec_col_2(mat.get_col(2).copy_to_vec());
        Vector<T> test_vec_col_2(
            TestBase::bundle,
            {static_cast<T>(3), static_cast<T>(7), static_cast<T>(11)}
        );
        ASSERT_VECTOR_EQ(vec_col_2, test_vec_col_2);

        Vector<T> vec_col_0_direct(mat.get_col(0));
        ASSERT_VECTOR_EQ(vec_col_0_direct, test_vec_col_0);

        Vector<T> vec_col_2_direct(mat.get_col(2));
        ASSERT_VECTOR_EQ(vec_col_2_direct, test_vec_col_2);

    }

    template <template <typename> typename M, typename T>
    void TestVectorToMatrixCol() {

        const M<T> const_mat(
            TestBase::bundle,
            {{static_cast<T>(1), static_cast<T>(2), static_cast<T>(3)},
             {static_cast<T>(4), static_cast<T>(5), static_cast<T>(6)},
             {static_cast<T>(7), static_cast<T>(8), static_cast<T>(9)},
             {static_cast<T>(10), static_cast<T>(11), static_cast<T>(12)}}
        );
        M<T> mat(const_mat);

        // Test assignment
        Vector<T> assign_vec(
            TestBase::bundle,
            {static_cast<T>(1), static_cast<T>(1), static_cast<T>(1), static_cast<T>(1)}
        );
        mat.get_col(2).set_from_vec(assign_vec);
        for (int j=0; j<2; ++j) {
            for (int i=0; i<4; ++i) {
                ASSERT_EQ(mat.get_elem(i, j), const_mat.get_elem(i, j));
            }
        }
        for (int i=0; i<4; ++i) {
            ASSERT_EQ(mat.get_elem(i, 2).get_scalar(), static_cast<T>(1));
        }

    }

    template <template <typename> typename M>
    void TestBadVectorToMatrixCol() {

        const M<double> const_mat(
            TestBase::bundle,
            {{1, 2, 3},
             {4, 5, 6},
             {7, 8, 9},
             {10, 11, 12}}
        );
        M<double> mat(const_mat);

        Vector<double> vec_too_small(
            TestBase::bundle,
            {1, 1, 1}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() mutable { mat.get_col(0).set_from_vec(vec_too_small); }
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() mutable { mat.get_col(1).set_from_vec(vec_too_small); }
        );

        Vector<double> vec_too_large(
            TestBase::bundle,
            {1, 1, 1, 1, 1, 1}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() mutable { mat.get_col(0).set_from_vec(vec_too_large);}
        );
        CHECK_FUNC_HAS_RUNTIME_ERROR(
            print_errors,
            [=]() mutable { mat.get_col(1).set_from_vec(vec_too_large); }
        );

    }

};

TEST_F(TypeConversion_Test, TestDenseToNoFillSparse) {
    TestDenseToNoFillSparse<__half>();
    TestDenseToNoFillSparse<float>();
    TestDenseToNoFillSparse<double>();
}

TEST_F(TypeConversion_Test, TestDenseToNoFillSparse_LONGRUNTIME) {
    TestDenseToNoFillSparse_LONGRUNTIME<__half>();
    TestDenseToNoFillSparse_LONGRUNTIME<float>();
    TestDenseToNoFillSparse_LONGRUNTIME<double>();
}

TEST_F(TypeConversion_Test, TestNoFillSparseToDense) {
    TestNoFillSparseToDense<__half>();
    TestNoFillSparseToDense<float>();
    TestNoFillSparseToDense<double>();
}

TEST_F(TypeConversion_Test, TestNoFillSparseBlockToDense) {
    TestNoFillSparseBlockToDense<__half>();
    TestNoFillSparseBlockToDense<float>();
    TestNoFillSparseBlockToDense<double>();
}

TEST_F(TypeConversion_Test, TestMatrixDenseColToVector) {
    TestMatrixColToVector<MatrixDense, __half>();
    TestMatrixColToVector<MatrixDense, float>();
    TestMatrixColToVector<MatrixDense, double>();
}

TEST_F(TypeConversion_Test, TestNoFillSparseColToVector) {
    TestMatrixColToVector<NoFillMatrixSparse, __half>();
    TestMatrixColToVector<NoFillMatrixSparse, float>();
    TestMatrixColToVector<NoFillMatrixSparse, double>();
}

TEST_F(TypeConversion_Test, TestVectorToMatrixDenseCol) {
    TestVectorToMatrixCol<MatrixDense, __half>();
    TestVectorToMatrixCol<MatrixDense, float>();
    TestVectorToMatrixCol<MatrixDense, double>();
}

TEST_F(TypeConversion_Test, TestBadVectorToMatrixDenseCol) {
    TestBadVectorToMatrixCol<MatrixDense>();
}