#include "../test.h"

// General matrix read tests
class MatrixRead_General_Test: public TestBase
{
public:

    template <template <typename> typename M>
    void ReadEmptyMatrix() {

        fs::path empty_file = read_matrix_dir / fs::path("empty.csv");
        M<double> test_empty(read_matrixCSV<M, double>(empty_file));
        ASSERT_EQ(test_empty.rows(), 0);
        ASSERT_EQ(test_empty.cols(), 0);

    }

    template <template <typename> typename M>
    void ReadBadMatrices() {

        // Try to load non-existent file
        fs::path bad_file_0 = read_matrix_dir / fs::path("thisfile");
        try {
            M<double> test(read_matrixCSV<M, double>(bad_file_0));
            FAIL();
        } catch (runtime_error e) { ; }

        // Try to load file with too small row
        fs::path bad_file_1 = read_matrix_dir / fs::path("bad1.csv");
        try {
            M<double> test(read_matrixCSV<M, double>(bad_file_1));
            FAIL();
        } catch (runtime_error e) { ;  }

        // Try to load file with too big rows
        fs::path bad_file_2 = read_matrix_dir / fs::path("bad2.csv");
        try {
            M<double> test(read_matrixCSV<M, double>(bad_file_2));
            FAIL();
        } catch (runtime_error e) { ; }

        // Try to load file with invalid character argument
        fs::path bad_file_3 = read_matrix_dir / fs::path("bad3.csv");
        try {
            M<double> test(read_matrixCSV<M, double>(bad_file_3));
            FAIL();
        } catch (runtime_error e) { ; }

    }

};

TEST_F(MatrixRead_General_Test, ReadEmptyMatrix_Both) {
    ReadEmptyMatrix<MatrixDense>();
    ReadEmptyMatrix<MatrixSparse>();
}

TEST_F(MatrixRead_General_Test, ReadBadFiles_Both) {
    ReadBadMatrices<MatrixDense>();
    ReadBadMatrices<MatrixSparse>();
}

template <typename T>
class MatrixRead_T_Test: public TestBase
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
    
        fs::path square1_file = read_matrix_dir / fs::path("square1.csv");
        fs::path square2_file = read_matrix_dir / fs::path("square2.csv");
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

        fs::path wide_file = read_matrix_dir / fs::path("wide.csv");
        fs::path tall_file = read_matrix_dir / fs::path("tall.csv");

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
        fs::path precise_file,
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
        fs::path precise_file,
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

// All type vector read tests
class MatrixRead_Vector_Test: public TestBase
{
public:

    template <typename T>
    void ReadVector(double u) {

        Matrix<T, Dynamic, 1> target {{1.}, {2.}, {3.}, {4.}, {5.}, {6.}};
        fs::path vector_file = read_matrix_dir / fs::path("vector.csv");
        MatrixVector<T> test(read_matrixCSV<MatrixVector, T>(vector_file));
        ASSERT_EQ(test.rows(), 6);
        ASSERT_EQ(test.cols(), 1);
        for (int i=0; i<6; ++i) { ASSERT_NEAR(static_cast<double>(test(i)), target(i), u); }

    }

};

TEST_F(MatrixRead_Vector_Test, ReadDoubleVector) { ReadVector<double>(u_dbl); }
TEST_F(MatrixRead_Vector_Test, ReadSingleVector) { ReadVector<double>(u_sgl); }
TEST_F(MatrixRead_Vector_Test, ReadHalfVector) { ReadVector<double>(u_hlf); }

TEST_F(MatrixRead_Vector_Test, FailOnMatrix) {    
    fs::path mat = read_matrix_dir / fs::path("square1.csv");
    try {
        MatrixVector<double> test(read_matrixCSV<MatrixVector, double>(mat));
        FAIL();
    } catch (runtime_error e) { cout << e.what() << endl; }
}

// Double type matrix read tests
class MatrixRead_Double_Test: public MatrixRead_T_Test<double> {};

TEST_F(MatrixRead_Double_Test, ReadSquareMatrix_Dense) { ReadSquareMatrix<MatrixDense>(u_dbl);}
TEST_F(MatrixRead_Double_Test, ReadSquareMatrix_Square) { ReadSquareMatrix<MatrixSparse>(u_dbl); }

TEST_F(MatrixRead_Double_Test, ReadWideTallMatrix_Dense) { ReadWideTallMatrix<MatrixDense>(u_dbl); }
TEST_F(MatrixRead_Double_Test, ReadWideTallMatrix_Sparse) { ReadWideTallMatrix<MatrixSparse>(u_dbl); }

TEST_F(MatrixRead_Double_Test, ReadPreciseMatrix_Both) {
    MatrixXd target_precise {
        {1.12345678901232, 1.12345678901234},
        {1.12345678901236, 1.12345678901238}
    };
    fs::path precise_file = read_matrix_dir / fs::path("double_precise.csv");
    ReadPrecise<MatrixDense>(target_precise, precise_file, u_dbl);
    ReadPrecise<MatrixSparse>(target_precise, precise_file, u_dbl);
}

TEST_F(MatrixRead_Double_Test, ReadDifferentThanPreciseMatrix_Both) {
    MatrixXd target_precise {
        {1.12345678901232, 1.12345678901234},
        {1.12345678901236, 1.12345678901238}
    };
    fs::path precise_file = read_matrix_dir / fs::path("double_precise.csv");
    ReadDifferentThanPrecise<MatrixDense>(target_precise, precise_file, u_dbl);
    ReadDifferentThanPrecise<MatrixSparse>(target_precise, precise_file, u_dbl);

}

TEST_F(MatrixRead_Double_Test, ReadPreciseMatrixDoubleLimit_Both) {
    MatrixXd target_precise {
        {1.1234567890123452, 1.1234567890123454},
        {1.1234567890123456, 1.1234567890123458}
    };
    fs::path precise_file = read_matrix_dir / fs::path("double_precise_manual.csv");
    ReadPrecise<MatrixDense>(target_precise, precise_file, u_dbl);
    ReadPrecise<MatrixSparse>(target_precise, precise_file, u_dbl);
}

TEST_F(MatrixRead_Double_Test, ReadDifferentThanPreciseMatrixDoubleLimit) {
    MatrixXd target_precise {
        {1.1234567890123452, 1.1234567890123454},
        {1.1234567890123456, 1.1234567890123458}
    };
    fs::path precise_file = read_matrix_dir / fs::path("double_precise_manual.csv");
    ReadDifferentThanPrecise<MatrixDense>(target_precise, precise_file, u_dbl);
    ReadDifferentThanPrecise<MatrixSparse>(target_precise, precise_file, u_dbl);
}

// Single type matrix read tests
class MatrixRead_Single_Test: public MatrixRead_T_Test<float> {};

TEST_F(MatrixRead_Single_Test, ReadSquareMatrix_Dense) { ReadSquareMatrix<MatrixDense>(u_sgl);}
TEST_F(MatrixRead_Single_Test, ReadSquareMatrix_Square) { ReadSquareMatrix<MatrixSparse>(u_sgl); }

TEST_F(MatrixRead_Single_Test, ReadWideTallMatrix_Dense) { ReadWideTallMatrix<MatrixDense>(u_sgl); }
TEST_F(MatrixRead_Single_Test, ReadWideTallMatrix_Sparse) { ReadWideTallMatrix<MatrixSparse>(u_sgl); }

TEST_F(MatrixRead_Single_Test, ReadPreciseMatrix_Both) {
    MatrixXf target_precise {
        {static_cast<float>(1.12345672), static_cast<float>(1.12345674)},
        {static_cast<float>(1.12345676), static_cast<float>(1.12345678)}
    };
    fs::path precise_file = read_matrix_dir / fs::path("single_precise.csv");
    ReadPrecise<MatrixDense>(target_precise, precise_file, u_sgl);
    ReadPrecise<MatrixSparse>(target_precise, precise_file, u_sgl);
}

TEST_F(MatrixRead_Single_Test, ReadDifferentThanPreciseMatrix_Both) {
    MatrixXf target_precise {
        {static_cast<float>(1.12345672), static_cast<float>(1.12345674)},
        {static_cast<float>(1.12345676), static_cast<float>(1.12345678)}
    };
    fs::path precise_file = read_matrix_dir / fs::path("single_precise.csv");
    ReadDifferentThanPrecise<MatrixDense>(target_precise, precise_file, u_sgl);
    ReadDifferentThanPrecise<MatrixSparse>(target_precise, precise_file, u_sgl);
}

// Half type matrix read tests
class MatrixRead_Half_Test: public MatrixRead_T_Test<half> {};

TEST_F(MatrixRead_Half_Test, ReadSquareMatrix_Dense) { ReadSquareMatrix<MatrixDense>(u_hlf);}
TEST_F(MatrixRead_Half_Test, ReadSquareMatrix_Square) { ReadSquareMatrix<MatrixSparse>(u_hlf); }

TEST_F(MatrixRead_Half_Test, ReadWideTallMatrix_Dense) { ReadWideTallMatrix<MatrixDense>(u_hlf); }
TEST_F(MatrixRead_Half_Test, ReadWideTallMatrix_Sparse) { ReadWideTallMatrix<MatrixSparse>(u_hlf); }

TEST_F(MatrixRead_Half_Test, ReadPreciseMatrix) {
    Matrix<half, Dynamic, Dynamic> target_precise {
        {static_cast<half>(1.123), static_cast<half>(1.124)},
        {static_cast<half>(1.125), static_cast<half>(1.126)}
    };
    fs::path precise_file = read_matrix_dir / fs::path("half_precise.csv");
    ReadPrecise<MatrixDense>(target_precise, precise_file, u_hlf);
    ReadPrecise<MatrixSparse>(target_precise, precise_file, u_hlf);
}

TEST_F(MatrixRead_Half_Test, ReadDifferentThanPreciseMatrix) {
    Matrix<half, Dynamic, Dynamic> target_precise {
        {static_cast<half>(1.123), static_cast<half>(1.124)},
        {static_cast<half>(1.125), static_cast<half>(1.126)}
    };
    fs::path precise_file = read_matrix_dir / fs::path("half_precise.csv");
    ReadDifferentThanPrecise<MatrixDense>(target_precise, precise_file, u_hlf);
    ReadDifferentThanPrecise<MatrixSparse>(target_precise, precise_file, u_hlf);
}
