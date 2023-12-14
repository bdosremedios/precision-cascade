#include "../test.h"

#include "types/types.h"

class TypeConversion_Test: public TestBase
{
public:

    template <typename T>
    void TestDenseToSparse() {

        // Test manual
        constexpr int m_manual(3);
        constexpr int n_manual(4);
        MatrixDense<T> dense_manual ({
            {static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4)},
            {static_cast<T>(5), static_cast<T>(6), static_cast<T>(7), static_cast<T>(8)},
            {static_cast<T>(9), static_cast<T>(10), static_cast<T>(11), static_cast<T>(12)}
        });
        MatrixSparse<T> sparse_manual(dense_manual.sparse());
        ASSERT_EQ(sparse_manual.rows(), m_manual);
        ASSERT_EQ(sparse_manual.cols(), n_manual);
        for (int i=0; i<m_manual; ++i) {
            for (int j=0; j<n_manual; ++j) {
                ASSERT_EQ(sparse_manual.coeff(i, j), dense_manual.coeff(i, j));
            }
        }

        // Test random
        constexpr int m_random(12);
        constexpr int n_random(7);
        MatrixDense<T> dense_rand(MatrixDense<T>::Random(12, 7));
        MatrixSparse<T> sparse_rand(dense_rand.sparse());
        ASSERT_EQ(sparse_rand.rows(), m_random);
        ASSERT_EQ(sparse_rand.cols(), n_random);
        for (int i=0; i<m_random; ++i) {
            for (int j=0; j<n_random; ++j) {
                ASSERT_EQ(sparse_rand.coeff(i, j), dense_rand.coeff(i, j));
            }
        }

    }

    template <typename T>
    void TestSparseBlockToDense() {

        MatrixSparse<T> const_mat ({
            {static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4), static_cast<T>(5)},
            {static_cast<T>(6), static_cast<T>(7), static_cast<T>(8), static_cast<T>(9), static_cast<T>(10)},
            {static_cast<T>(11), static_cast<T>(12), static_cast<T>(13), static_cast<T>(14), static_cast<T>(15)},
            {static_cast<T>(16), static_cast<T>(17), static_cast<T>(18), static_cast<T>(19), static_cast<T>(20)}
        });
        MatrixSparse<T> mat(const_mat);
        
        // Test cast/access for block 0, 0, 3, 4
        MatrixDense<T> mat_0_0_3_4(mat.block(0, 0, 3, 4));
        MatrixDense<T> test_0_0_3_4 ({
            {static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4)},
            {static_cast<T>(6), static_cast<T>(7), static_cast<T>(8), static_cast<T>(9)},
            {static_cast<T>(11), static_cast<T>(12), static_cast<T>(13), static_cast<T>(14)}
        });
        ASSERT_MATRIX_EQ(mat_0_0_3_4, test_0_0_3_4);

        // Test cast/access for block 1, 2, 3, 1
        MatrixDense<T> mat_1_2_3_1(mat.block(1, 2, 3, 1));
        MatrixDense<T> test_1_2_3_1 ({
            {static_cast<T>(8)},
            {static_cast<T>(13)},
            {static_cast<T>(18)}
        });
        ASSERT_MATRIX_EQ(mat_1_2_3_1, test_1_2_3_1);

    }

    template <template <typename> typename M, typename T>
    void TestMatrixColToMatrixVector() {

        M<T> mat ({
            {static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4)},
            {static_cast<T>(5), static_cast<T>(6), static_cast<T>(7), static_cast<T>(8)},
            {static_cast<T>(9), static_cast<T>(10), static_cast<T>(11), static_cast<T>(12)}
        });

        MatrixVector<T> vec_col_0(mat.col(0));
        MatrixVector<T> test_vec_col_0 ({static_cast<T>(1),
                                         static_cast<T>(5),
                                         static_cast<T>(9)});
        ASSERT_VECTOR_EQ(vec_col_0, test_vec_col_0);

        MatrixVector<T> vec_col_2(mat.col(2));
        MatrixVector<T> test_vec_col_2 ({static_cast<T>(3),
                                         static_cast<T>(7),
                                         static_cast<T>(11)});
        ASSERT_VECTOR_EQ(vec_col_2, test_vec_col_2);

    }

    template <template <typename> typename M, typename T>
    void TestMatrixVectorToMatrixCol() {

        const M<T> const_mat ({
            {static_cast<T>(1), static_cast<T>(2), static_cast<T>(3)},
            {static_cast<T>(4), static_cast<T>(5), static_cast<T>(6)},
            {static_cast<T>(7), static_cast<T>(8), static_cast<T>(9)},
            {static_cast<T>(10), static_cast<T>(11), static_cast<T>(12)}
        });
        M<T> mat(const_mat);

        // Test assignment
        MatrixVector<T> assign_vec ({static_cast<T>(1),
                                     static_cast<T>(1),
                                     static_cast<T>(1),
                                     static_cast<T>(1)});
        mat.col(2) = assign_vec;
        for (int j=0; j<2; ++j) {
            for (int i=0; i<4; ++i) {
                ASSERT_EQ(mat.coeff(i, j), const_mat.coeff(i, j));
            }
        }
        for (int i=0; i<4; ++i) { ASSERT_EQ(mat.coeff(i, 2), static_cast<T>(1)); }

    }

};

TEST_F(TypeConversion_Test, TestDenseToSparse) {
    TestDenseToSparse<half>();
    TestDenseToSparse<float>();
    TestDenseToSparse<double>();
}

TEST_F(TypeConversion_Test, TestSparseBlockToDense) {
    TestSparseBlockToDense<half>();
    TestSparseBlockToDense<float>();
    TestSparseBlockToDense<double>();
}

TEST_F(TypeConversion_Test, TestDenseColToMatrixVector) {
    TestMatrixColToMatrixVector<MatrixDense, half>();
    TestMatrixColToMatrixVector<MatrixDense, float>();
    TestMatrixColToMatrixVector<MatrixDense, double>();
}

TEST_F(TypeConversion_Test, TestSparseColToMatrixVector) {
    TestMatrixColToMatrixVector<MatrixSparse, half>();
    TestMatrixColToMatrixVector<MatrixSparse, float>();
    TestMatrixColToMatrixVector<MatrixSparse, double>();
}

TEST_F(TypeConversion_Test, TestMatrixVectorToDenseCol) {
    TestMatrixVectorToMatrixCol<MatrixDense, half>();
    TestMatrixVectorToMatrixCol<MatrixDense, float>();
    TestMatrixVectorToMatrixCol<MatrixDense, double>();
}

TEST_F(TypeConversion_Test, TestMatrixVectorToSparseCol) {
    TestMatrixVectorToMatrixCol<MatrixSparse, half>();
    TestMatrixVectorToMatrixCol<MatrixSparse, float>();
    TestMatrixVectorToMatrixCol<MatrixSparse, double>();
}