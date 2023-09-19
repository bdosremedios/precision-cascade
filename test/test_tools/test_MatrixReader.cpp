#include "../test.h"

// General matrix read tests

class MatrixReadGeneralTest: public TestBase {

public:

    template <template <typename> typename M>
    void ReadEmptyMatrix() {

        string empty_file = read_matrix_dir + "empty.csv";
        M<double> test_empty(read_matrixCSV<M, double>(empty_file));
        ASSERT_EQ(test_empty.rows(), 0);
        ASSERT_EQ(test_empty.cols(), 0);

    }

    template <template <typename> typename M>
    void ReadBadMatrices() {

        // Try to load non-existent file
        string bad_file_0 = read_matrix_dir + "thisfile";
        try {
            M<double> test(read_matrixCSV<M, double>(bad_file_0));
            FAIL();
        } catch (runtime_error e) {
            EXPECT_EQ(
                e.what(),
                "Failed to read: " + bad_file_0
            );
        }

        // Try to load file with too small row
        string bad_file_1 = read_matrix_dir + "bad1.csv";
        try {
            M<double> test(read_matrixCSV<M, double>(bad_file_1));
            FAIL();
        } catch (runtime_error e) {
            EXPECT_EQ(
                e.what(),
                "Error in: " + bad_file_1 + "\n" + "Row 3 does not meet column size of 3"
            );
        }

        // Try to load file with too big rows
        string bad_file_2 = read_matrix_dir + "bad2.csv";
        try {
            M<double> test(read_matrixCSV<M, double>(bad_file_2));
            FAIL();
        } catch (runtime_error e) {
            EXPECT_EQ(
                e.what(),
                "Error in: " + bad_file_2 + "\n" + "Row 2 exceeds column size of 3"
            );
        }

        // Try to load file with invalid character argument
        string bad_file_3 = read_matrix_dir + "bad3.csv";
        try {
            M<double> test(read_matrixCSV<M, double>(bad_file_3));
            FAIL();
        } catch (runtime_error e) {
            EXPECT_EQ(
                e.what(),
                "Error in: " + bad_file_3 + "\n" + "Invalid argument in file, failed to convert to numeric"
            );
        }

    }

};

TEST_F(MatrixReadGeneralTest, ReadEmptyMatrix_Both) {
    ReadEmptyMatrix<MatrixDense>();
    ReadEmptyMatrix<MatrixSparse>();
}

TEST_F(MatrixReadGeneralTest, ReadBadFiles_Both) {
    ReadBadMatrices<MatrixDense>();
    ReadBadMatrices<MatrixSparse>();
}

template <typename T>
class MatrixReadTTest: public TestBase
{
public:

    template <template <typename> typename M>
    void ReadSquareMatrix(double u) {

        Matrix<double, Dynamic, Dynamic> temp_target1 {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        };
        Matrix<T, Dynamic, Dynamic> target1 = temp_target1.template cast<T>();

        Matrix<double, Dynamic, Dynamic> temp_target2 {
            {1, 2, 3, 4, 5},
            {6, 7, 8, 9, 10},
            {11, 12, 13, 14, 15},
            {16, 17, 18, 19, 20},
            {21, 22, 23, 24, 25}
        };
        Matrix<T, Dynamic, Dynamic> target2 = temp_target2.template cast<T>();
    
        string square1_file = read_matrix_dir + "square1.csv";
        string square2_file = read_matrix_dir + "square2.csv";
        M<T> test1(read_matrixCSV<M, T>(square1_file));
        M<T> test2(read_matrixCSV<M, T>(square2_file));

        // Check that read is correct for first file
        ASSERT_EQ(test1.rows(), 3);
        ASSERT_EQ(test1.cols(), 3);
        for (int i=0; i<3; ++i) {
            for (int j=0; j<3; ++j) {
                ASSERT_NEAR(test1.coeff(i, j), target1(i, j), u);
            }
        }

        // Check that read is correct for second file
        ASSERT_EQ(test2.rows(), 5);
        ASSERT_EQ(test2.cols(), 5);
        for (int i=0; i<5; ++i) {
            for (int j=0; j<5; ++j) {
                ASSERT_NEAR(test2.coeff(i, j), target2(i, j), u_dbl);
            }
        }

    }

    template <template <typename> typename M>
    void ReadWideTallMatrix(double u) {

        Matrix<double, Dynamic, Dynamic> temp_target_wide {
            {10, 9, 8, 7, 6},
            {5, 4, 3, 2, 1}
        };
        Matrix<T, Dynamic, Dynamic> target_wide = temp_target_wide.template cast<T>();

        Matrix<double, Dynamic, Dynamic> temp_target_tall {
            {1, 2},
            {3, 4},
            {5, 6},
            {7, 8}
        };
        Matrix<T, Dynamic, Dynamic> target_tall = temp_target_tall.template cast<T>();

        string wide_file = read_matrix_dir + "wide.csv";
        string tall_file = read_matrix_dir + "tall.csv";

        M<T> test_wide(read_matrixCSV<M, T>(wide_file));
        M<T> test_tall(read_matrixCSV<M, T>(tall_file));

        // Check that read is correct for first file
        ASSERT_EQ(test_wide.rows(), 2);
        ASSERT_EQ(test_wide.cols(), 5);
        for (int i=0; i<2; ++i) {
            for (int j=0; j<5; ++j) {
                ASSERT_NEAR(test_wide.coeffRef(i, j), target_wide(i, j), u);
            }
        }

        // Check that read is correct for second file
        ASSERT_EQ(test_tall.rows(), 4);
        ASSERT_EQ(test_tall.cols(), 2);
        for (int i=0; i<4; ++i) {
            for (int j=0; j<2; ++j) {
                ASSERT_NEAR(test_tall.coeffRef(i, j), target_tall(i, j), u);
            }
        }

    }

    template <template <typename> typename M>
    void ReadPrecise(
        Matrix<T, Dynamic, Dynamic> target_precise,
        string precise_file,
        double u
    ) {

        M<T> test_precise(read_matrixCSV<M, T>(precise_file));

        ASSERT_EQ(test_precise.rows(), 2);
        ASSERT_EQ(test_precise.cols(), 2);
        for (int i=0; i<2; ++i) {
            for (int j=0; j<2; ++j) {
                ASSERT_NEAR(test_precise.coeffRef(i, j), target_precise(i, j), u);
            }
        }

    }

    template <template <typename> typename M>
    void ReadDifferentThanPrecise(
        Matrix<T, Dynamic, Dynamic> target_precise,
        string precise_file,
        double u
    ) {

        T eps = static_cast<T>(1.5*u);
        Matrix<T, Dynamic, Dynamic> miss_precise_up = target_precise + eps*Matrix<T, Dynamic, Dynamic>::Ones(2, 2);
        Matrix<T, Dynamic, Dynamic> miss_precise_down = target_precise - eps*Matrix<T, Dynamic, Dynamic>::Ones(2, 2);

        M<T> test_precise(read_matrixCSV<M, T>(precise_file));

        ASSERT_EQ(test_precise.rows(), 2);
        ASSERT_EQ(test_precise.cols(), 2);
        for (int i=0; i<2; ++i) {
            for (int j=0; j<2; ++j) {
                ASSERT_LT(test_precise.coeffRef(i, j), miss_precise_up(i, j));
                ASSERT_GT(test_precise.coeffRef(i, j), miss_precise_down(i, j));
            }
        }

    }

};

// Double type matrix read tests
class MatrixReadDoubleTest: public MatrixReadTTest<double> {};

TEST_F(MatrixReadDoubleTest, ReadSquareMatrix_Dense) { ReadSquareMatrix<MatrixDense>(u_dbl);}
TEST_F(MatrixReadDoubleTest, ReadSquareMatrix_Square) { ReadSquareMatrix<MatrixSparse>(u_dbl); }

TEST_F(MatrixReadDoubleTest, ReadWideTallMatrix_Dense) { ReadWideTallMatrix<MatrixDense>(u_dbl); }
TEST_F(MatrixReadDoubleTest, ReadWideTallMatrix_Sparse) { ReadWideTallMatrix<MatrixSparse>(u_dbl); }

TEST_F(MatrixReadDoubleTest, ReadPreciseMatrix_Both) {
    MatrixXd target_precise {
        {1.12345678901232, 1.12345678901234},
        {1.12345678901236, 1.12345678901238}
    };
    string precise_file = read_matrix_dir + "double_precise.csv";
    ReadPrecise<MatrixDense>(target_precise, precise_file, u_dbl);
    ReadPrecise<MatrixSparse>(target_precise, precise_file, u_dbl);
}

TEST_F(MatrixReadDoubleTest, ReadDifferentThanPreciseMatrix_Both) {
    MatrixXd target_precise {
        {1.12345678901232, 1.12345678901234},
        {1.12345678901236, 1.12345678901238}
    };
    string precise_file = read_matrix_dir + "double_precise.csv";
    ReadDifferentThanPrecise<MatrixDense>(target_precise, precise_file, u_dbl);
    ReadDifferentThanPrecise<MatrixSparse>(target_precise, precise_file, u_dbl);

}

TEST_F(MatrixReadDoubleTest, ReadPreciseMatrixDoubleLimit_Both) {
    MatrixXd target_precise {
        {1.1234567890123452, 1.1234567890123454},
        {1.1234567890123456, 1.1234567890123458}
    };
    string precise_file = read_matrix_dir + "double_precise_manual.csv";
    ReadPrecise<MatrixDense>(target_precise, precise_file, u_dbl);
    ReadPrecise<MatrixSparse>(target_precise, precise_file, u_dbl);
}

TEST_F(MatrixReadDoubleTest, ReadDifferentThanPreciseMatrixDoubleLimit) {
    MatrixXd target_precise {
        {1.1234567890123452, 1.1234567890123454},
        {1.1234567890123456, 1.1234567890123458}
    };
    string precise_file = read_matrix_dir + "double_precise_manual.csv";
    ReadDifferentThanPrecise<MatrixDense>(target_precise, precise_file, u_dbl);
    ReadDifferentThanPrecise<MatrixSparse>(target_precise, precise_file, u_dbl);
}

// Single type matrix read tests
class MatrixReadSingleTest: public MatrixReadTTest<float> {};

TEST_F(MatrixReadSingleTest, ReadSquareMatrix_Dense) { ReadSquareMatrix<MatrixDense>(u_sgl);}
TEST_F(MatrixReadSingleTest, ReadSquareMatrix_Square) { ReadSquareMatrix<MatrixSparse>(u_sgl); }

TEST_F(MatrixReadSingleTest, ReadWideTallMatrix_Dense) { ReadWideTallMatrix<MatrixDense>(u_sgl); }
TEST_F(MatrixReadSingleTest, ReadWideTallMatrix_Sparse) { ReadWideTallMatrix<MatrixSparse>(u_sgl); }

TEST_F(MatrixReadSingleTest, ReadPreciseMatrix_Both) {
    MatrixXf target_precise {
        {static_cast<float>(1.12345672), static_cast<float>(1.12345674)},
        {static_cast<float>(1.12345676), static_cast<float>(1.12345678)}
    };
    string precise_file = read_matrix_dir + "single_precise.csv";
    ReadPrecise<MatrixDense>(target_precise, precise_file, u_sgl);
    ReadPrecise<MatrixSparse>(target_precise, precise_file, u_sgl);
}

TEST_F(MatrixReadSingleTest, ReadDifferentThanPreciseMatrix_Both) {
    MatrixXf target_precise {
        {static_cast<float>(1.12345672), static_cast<float>(1.12345674)},
        {static_cast<float>(1.12345676), static_cast<float>(1.12345678)}
    };
    string precise_file = read_matrix_dir + "single_precise.csv";
    ReadDifferentThanPrecise<MatrixDense>(target_precise, precise_file, u_sgl);
    ReadDifferentThanPrecise<MatrixSparse>(target_precise, precise_file, u_sgl);
}

// Half type matrix read tests
class MatrixReadHalfTest: public MatrixReadTTest<half> {};

TEST_F(MatrixReadHalfTest, ReadSquareMatrix_Dense) { ReadSquareMatrix<MatrixDense>(u_hlf);}
TEST_F(MatrixReadHalfTest, ReadSquareMatrix_Square) { ReadSquareMatrix<MatrixSparse>(u_hlf); }

TEST_F(MatrixReadHalfTest, ReadWideTallMatrix_Dense) { ReadWideTallMatrix<MatrixDense>(u_hlf); }
TEST_F(MatrixReadHalfTest, ReadWideTallMatrix_Sparse) { ReadWideTallMatrix<MatrixSparse>(u_hlf); }

TEST_F(MatrixReadHalfTest, ReadPreciseMatrix) {
    Matrix<half, Dynamic, Dynamic> target_precise {
        {static_cast<half>(1.123), static_cast<half>(1.124)},
        {static_cast<half>(1.125), static_cast<half>(1.126)}
    };
    string precise_file = read_matrix_dir + "half_precise.csv";
    ReadPrecise<MatrixDense>(target_precise, precise_file, u_hlf);
    ReadPrecise<MatrixSparse>(target_precise, precise_file, u_hlf);
}

TEST_F(MatrixReadHalfTest, ReadDifferentThanPreciseMatrix) {
    Matrix<half, Dynamic, Dynamic> target_precise {
        {static_cast<half>(1.123), static_cast<half>(1.124)},
        {static_cast<half>(1.125), static_cast<half>(1.126)}
    };
    string precise_file = read_matrix_dir + "half_precise.csv";
    ReadDifferentThanPrecise<MatrixDense>(target_precise, precise_file, u_hlf);
    ReadDifferentThanPrecise<MatrixSparse>(target_precise, precise_file, u_hlf);
}
