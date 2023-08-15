#include "gtest/gtest.h"
#include "Eigen/Dense"

#include "test.h"

#include "read_matrix/MatrixReader.h"

#include <string>
#include <cmath>

using read_matrix::read_matrix_csv;
using Eigen::half;
using Eigen::MatrixXd, Eigen::MatrixXf;

using std::string;
using std::cout, std::endl;
using std::runtime_error;
using std::pow;

// General matrix read tests

class MatrixReadGeneralTest: public TestBase {};

TEST_F(MatrixReadGeneralTest, ReadEmptyMatrix) {

    string empty_file = read_matrix_dir + "empty.csv";
    MatrixXd test_empty(read_matrix_csv<double>(empty_file));
    ASSERT_EQ(test_empty.rows(), 0);
    ASSERT_EQ(test_empty.cols(), 0);

}

TEST_F(MatrixReadGeneralTest, ReadBadFiles) {

    // Try to load non-existent file
    string bad_file_0 = read_matrix_dir + "thisfile";
    try {
        MatrixXd test(read_matrix_csv<double>(bad_file_0));
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
        MatrixXd test(read_matrix_csv<double>(bad_file_1));
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
        MatrixXd test(read_matrix_csv<double>(bad_file_2));
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
        MatrixXd test(read_matrix_csv<double>(bad_file_3));
        FAIL();
    } catch (runtime_error e) {
        EXPECT_EQ(
            e.what(),
            "Error in: " + bad_file_3 + "\n" + "Invalid argument in file, failed to convert to numeric"
        );
    }
    
}

// Double type matrix read tests
class MatrixReadDoubleTest: public TestBase {};

TEST_F(MatrixReadDoubleTest, ReadSquareMatrix) {

    MatrixXd target1 {{1, 2, 3},
                      {4, 5, 6},
                      {7, 8, 9}};
    MatrixXd target2 {{1, 2, 3, 4, 5},
                      {6, 7, 8, 9, 10},
                      {11, 12, 13, 14, 15},
                      {16, 17, 18, 19, 20},
                      {21, 22, 23, 24, 25}};
    string square1_file = read_matrix_dir + "square1.csv";
    string square2_file = read_matrix_dir + "square2.csv";
    MatrixXd test1(read_matrix_csv<double>(square1_file));
    MatrixXd test2(read_matrix_csv<double>(square2_file));

    // Check that read is correct for first file
    ASSERT_EQ(test1.rows(), 3);
    ASSERT_EQ(test1.cols(), 3);
    for (int i=0; i<3; ++i) {
        for (int j=0; j<3; ++j) {
            ASSERT_NEAR(test1(i, j), target1(i, j), u_dbl);
        }
    }

    // Check that read is correct for second file
    ASSERT_EQ(test2.rows(), 5);
    ASSERT_EQ(test2.cols(), 5);
    for (int i=0; i<5; ++i) {
        for (int j=0; j<5; ++j) {
            ASSERT_NEAR(test2(i, j), target2(i, j), u_dbl);
        }
    }

}

TEST_F(MatrixReadDoubleTest, ReadWideTallMatrix) {

    MatrixXd target_wide {{10, 9, 8, 7, 6},
                          {5, 4, 3, 2, 1}};
    MatrixXd target_tall {{1, 2},
                          {3, 4},
                          {5, 6},
                          {7, 8}};
    string wide_file = read_matrix_dir + "wide.csv";
    string tall_file = read_matrix_dir + "tall.csv";
    MatrixXd test_wide(read_matrix_csv<double>(wide_file));
    MatrixXd test_tall(read_matrix_csv<double>(tall_file));

    // Check that read is correct for first file
    ASSERT_EQ(test_wide.rows(), 2);
    ASSERT_EQ(test_wide.cols(), 5);
    for (int i=0; i<2; ++i) {
        for (int j=0; j<5; ++j) {
            EXPECT_NEAR(test_wide(i, j), target_wide(i, j), u_dbl);
        }
    }

    // Check that read is correct for second file
    ASSERT_EQ(test_tall.rows(), 4);
    ASSERT_EQ(test_tall.cols(), 2);
    for (int i=0; i<4; ++i) {
        for (int j=0; j<2; ++j) {
            ASSERT_NEAR(test_tall(i, j), target_tall(i, j), u_dbl);
        }
    }
    
}

TEST_F(MatrixReadDoubleTest, ReadPreciseMatrix) {

    MatrixXd target_precise {
        {1.12345678901232, 1.12345678901234},
        {1.12345678901236, 1.12345678901238}
    };
    string precise_file = read_matrix_dir + "double_precise.csv";
    MatrixXd test_precise(read_matrix_csv<double>(precise_file));

    ASSERT_EQ(test_precise.rows(), 2);
    ASSERT_EQ(test_precise.cols(), 2);
    for (int i=0; i<2; ++i) {
        for (int j=0; j<2; ++j) {
            ASSERT_NEAR(test_precise(i, j), target_precise(i, j), u_dbl);
        }
    }

}

TEST_F(MatrixReadDoubleTest, ReadDifferentThanPreciseMatrix) {

    double eps = 1.5*pow(10, -14);
    MatrixXd miss_precise_up {
        {1.12345678901232+eps, 1.12345678901234+eps},
        {1.12345678901236+eps, 1.12345678901238+eps}
    };
    MatrixXd miss_precise_down {
        {1.12345678901232-eps, 1.12345678901234-eps},
        {1.12345678901236-eps, 1.12345678901238-eps}
    };
    string precise_file = read_matrix_dir + "double_precise.csv";
    MatrixXd test_precise(read_matrix_csv<double>(precise_file));

    ASSERT_EQ(test_precise.rows(), 2);
    ASSERT_EQ(test_precise.cols(), 2);
    for (int i=0; i<2; ++i) {
        for (int j=0; j<2; ++j) {
            ASSERT_LT(test_precise(i, j), miss_precise_up(i, j));
            ASSERT_GT(test_precise(i, j), miss_precise_down(i, j));
        }
    }

}

TEST_F(MatrixReadDoubleTest, ReadPreciseMatrixDoubleLimit) {

    MatrixXd target_precise {
        {1.1234567890123452, 1.1234567890123454},
        {1.1234567890123456, 1.1234567890123458}
    };
    string precise_file = read_matrix_dir + "double_precise_manual.csv";
    MatrixXd test_precise(read_matrix_csv<double>(precise_file));

    ASSERT_EQ(test_precise.rows(), 2);
    ASSERT_EQ(test_precise.cols(), 2);
    for (int i=0; i<2; ++i) {
        for (int j=0; j<2; ++j) {
            ASSERT_NEAR(test_precise(i, j), target_precise(i, j), u_dbl);
        }
    }

}

TEST_F(MatrixReadDoubleTest, ReadDifferentThanPreciseMatrixDoubleLimit) {

    double eps = 1.5*u_dbl;
    MatrixXd miss_precise_up {
        {1.1234567890123452+eps, 1.1234567890123454+eps},
        {1.1234567890123456+eps, 1.1234567890123458+eps}
    };
    MatrixXd miss_precise_down {
        {1.1234567890123452-eps, 1.1234567890123454-eps},
        {1.1234567890123456-eps, 1.1234567890123458-eps}
    };
    string precise_file = read_matrix_dir + "double_precise_manual.csv";
    MatrixXd test_precise(read_matrix_csv<double>(precise_file));

    ASSERT_EQ(test_precise.rows(), 2);
    ASSERT_EQ(test_precise.cols(), 2);
    for (int i=0; i<2; ++i) {
        for (int j=0; j<2; ++j) {
            ASSERT_LT(test_precise(i, j), miss_precise_up(i, j));
            ASSERT_GT(test_precise(i, j), miss_precise_down(i, j));
        }
    }

}

// Single type matrix read tests
class MatrixReadSingleTest: public TestBase {};

TEST_F(MatrixReadSingleTest, ReadSquareMatrix) {

    MatrixXf target1 {{1, 2, 3},
                      {4, 5, 6},
                      {7, 8, 9}};
    MatrixXf target2 {{1, 2, 3, 4, 5},
                      {6, 7, 8, 9, 10},
                      {11, 12, 13, 14, 15},
                      {16, 17, 18, 19, 20},
                      {21, 22, 23, 24, 25}};
    string square1_file = read_matrix_dir + "square1.csv";
    string square2_file = read_matrix_dir + "square2.csv";
    MatrixXf test1(read_matrix_csv<float>(square1_file));
    MatrixXf test2(read_matrix_csv<float>(square2_file));

    // Check that read is correct for first file
    ASSERT_EQ(test1.rows(), 3);
    ASSERT_EQ(test1.cols(), 3);
    for (int i=0; i<3; ++i) {
        for (int j=0; j<3; ++j) {
            ASSERT_NEAR(test1(i, j), target1(i, j), u_sgl);
        }
    }

    // Check that read is correct for second file
    ASSERT_EQ(test2.rows(), 5);
    ASSERT_EQ(test2.cols(), 5);
    for (int i=0; i<5; ++i) {
        for (int j=0; j<5; ++j) {
            ASSERT_NEAR(test2(i, j), target2(i, j), u_sgl);
        }
    }

}

TEST_F(MatrixReadSingleTest, ReadWideTallMatrix) {

    MatrixXf target_wide {{10, 9, 8, 7, 6},
                          {5, 4, 3, 2, 1}};
    MatrixXf target_tall {{1, 2},
                          {3, 4},
                          {5, 6},
                          {7, 8}};
    string wide_file = read_matrix_dir + "wide.csv";
    string tall_file = read_matrix_dir + "tall.csv";
    MatrixXf test_wide(read_matrix_csv<float>(wide_file));
    MatrixXf test_tall(read_matrix_csv<float>(tall_file));

    // Check that read is correct for first file
    ASSERT_EQ(test_wide.rows(), 2);
    ASSERT_EQ(test_wide.cols(), 5);
    for (int i=0; i<2; ++i) {
        for (int j=0; j<5; ++j) {
            ASSERT_NEAR(test_wide(i, j), target_wide(i, j), u_sgl);
        }
    }

    // Check that read is correct for second file
    ASSERT_EQ(test_tall.rows(), 4);
    ASSERT_EQ(test_tall.cols(), 2);
    for (int i=0; i<4; ++i) {
        for (int j=0; j<2; ++j) {
            ASSERT_NEAR(test_tall(i, j), target_tall(i, j), u_sgl);
        }
    }
    
}

TEST_F(MatrixReadSingleTest, ReadPreciseMatrix) {

    MatrixXf target_precise {{static_cast<float>(1.12345672), static_cast<float>(1.12345674)},
                              {static_cast<float>(1.12345676), static_cast<float>(1.12345678)}};
    string precise_file = read_matrix_dir + "single_precise.csv";
    MatrixXf test_precise(read_matrix_csv<float>(precise_file));

    ASSERT_EQ(test_precise.rows(), 2);
    ASSERT_EQ(test_precise.cols(), 2);
    for (int i=0; i<2; ++i) {
        for (int j=0; j<2; ++j) {
            ASSERT_NEAR(test_precise(i, j), target_precise(i, j), u_sgl);
        }
    }

}

TEST_F(MatrixReadSingleTest, ReadDifferentThanPreciseMatrix) {
    
    float eps = static_cast<float>(1.5*u_sgl);
    MatrixXf miss_precise_up {
        {static_cast<float>(1.12345672)+eps, static_cast<float>(1.12345674)+eps},
        {static_cast<float>(1.12345676)+eps, static_cast<float>(1.12345678)+eps}
    };
    MatrixXf miss_precise_down {
        {static_cast<float>(1.12345672)-eps, static_cast<float>(1.12345674)-eps},
        {static_cast<float>(1.12345676)-eps, static_cast<float>(1.12345678)-eps}
    };
    string precise_file = read_matrix_dir + "single_precise.csv";
    MatrixXf test_precise(read_matrix_csv<float>(precise_file));

    ASSERT_EQ(test_precise.rows(), 2);
    ASSERT_EQ(test_precise.cols(), 2);
    for (int i=0; i<2; ++i) {
        for (int j=0; j<2; ++j) {
            ASSERT_LT(test_precise(i, j), miss_precise_up(i, j));
            ASSERT_GT(test_precise(i, j), miss_precise_down(i, j));
        }
    }

}

// Half type matrix read tests
class MatrixReadHalfTest: public TestBase {};

TEST_F(MatrixReadHalfTest, ReadSquareMatrix) {

    Matrix<double, Dynamic, Dynamic> temp1 {{1, 2, 3},
                                            {4, 5, 6},
                                            {7, 8, 9}};
    Matrix<half, Dynamic, Dynamic> target1 = temp1.cast<half>();
    Matrix<double, Dynamic, Dynamic> temp2 {{1, 2, 3, 4, 5},
                                            {6, 7, 8, 9, 10},
                                            {11, 12, 13, 14, 15},
                                            {16, 17, 18, 19, 20},
                                            {21, 22, 23, 24, 25}};
    Matrix<half, Dynamic, Dynamic> target2 = temp2.cast<half>();
    string square1_file = read_matrix_dir + "square1.csv";
    string square2_file = read_matrix_dir + "square2.csv";

    Matrix<half, Dynamic, Dynamic> test1(read_matrix_csv<half>(square1_file));
    Matrix<half, Dynamic, Dynamic> test2(read_matrix_csv<half>(square2_file));

    // Check that read is correct for first file
    ASSERT_EQ(test1.rows(), 3);
    ASSERT_EQ(test1.cols(), 3);
    for (int i=0; i<3; ++i) {
        for (int j=0; j<3; ++j) {
            ASSERT_NEAR(test1(i, j), target1(i, j), u_hlf);
        }
    }

    // Check that read is correct for second file
    ASSERT_EQ(test2.rows(), 5);
    ASSERT_EQ(test2.cols(), 5);
    for (int i=0; i<5; ++i) {
        for (int j=0; j<5; ++j) {
            ASSERT_NEAR(test2(i, j), target2(i, j), u_hlf);
        }
    }

}

TEST_F(MatrixReadHalfTest, ReadWideTallMatrix) {

    Matrix<double, Dynamic, Dynamic> temp_wide {{10, 9, 8, 7, 6},
                                                {5, 4, 3, 2, 1}};
    Matrix<half, Dynamic, Dynamic> target_wide = temp_wide.cast<half>();
    Matrix<double, Dynamic, Dynamic> temp_tall {{1, 2},
                                                {3, 4},
                                                {5, 6},
                                                {7, 8}};
    Matrix<half, Dynamic, Dynamic> target_tall = temp_tall.cast<half>();
    string wide_file = read_matrix_dir + "wide.csv";
    string tall_file = read_matrix_dir + "tall.csv";
    Matrix<half, Dynamic, Dynamic> test_wide(read_matrix_csv<half>(wide_file));
    Matrix<half, Dynamic, Dynamic> test_tall(read_matrix_csv<half>(tall_file));

    // Check that read is correct for first file
    ASSERT_EQ(test_wide.rows(), 2);
    ASSERT_EQ(test_wide.cols(), 5);
    for (int i=0; i<2; ++i) {
        for (int j=0; j<5; ++j) {
            ASSERT_NEAR(test_wide(i, j), target_wide(i, j), u_hlf);
        }
    }

    // Check that read is correct for second file
    ASSERT_EQ(test_tall.rows(), 4);
    ASSERT_EQ(test_tall.cols(), 2);
    for (int i=0; i<4; ++i) {
        for (int j=0; j<2; ++j) {
            ASSERT_NEAR(test_tall(i, j), target_tall(i, j), u_hlf);
        }
    }
    
}

TEST_F(MatrixReadHalfTest, ReadPreciseMatrix) {

    Matrix<half, Dynamic, Dynamic> target_precise {
        {static_cast<half>(1.123), static_cast<half>(1.124)},
        {static_cast<half>(1.125), static_cast<half>(1.126)}
    };
    string precise_file = read_matrix_dir + "half_precise.csv";
    Matrix<half, Dynamic, Dynamic> test_precise(read_matrix_csv<half>(precise_file));

    ASSERT_EQ(test_precise.rows(), 2);
    ASSERT_EQ(test_precise.cols(), 2);
    for (int i=0; i<2; ++i) {
        for (int j=0; j<2; ++j) {
            ASSERT_NEAR(test_precise(i, j), target_precise(i, j), u_hlf);
        }
    }

}

TEST_F(MatrixReadHalfTest, ReadDifferentThanPreciseMatrix) {
    
    half eps = static_cast<half>(1.5*u_hlf);
    Matrix<half, Dynamic, Dynamic> miss_precise_up {
        {static_cast<half>(1.123)+eps, static_cast<half>(1.124)+eps},
        {static_cast<half>(1.125)+eps, static_cast<half>(1.126)+eps}
    };
    Matrix<half, Dynamic, Dynamic> miss_precise_down {
        {static_cast<half>(1.123)-eps, static_cast<half>(1.124)-eps},
        {static_cast<half>(1.125)-eps, static_cast<half>(1.126)-eps}
    };
    string precise_file = read_matrix_dir + "half_precise.csv";
    Matrix<half, Dynamic, Dynamic> test_precise(read_matrix_csv<half>(precise_file));

    ASSERT_EQ(test_precise.rows(), 2);
    ASSERT_EQ(test_precise.cols(), 2);
    for (int i=0; i<2; ++i) {
        for (int j=0; j<2; ++j) {
            ASSERT_LT(test_precise(i, j), miss_precise_up(i, j));
            ASSERT_GT(test_precise(i, j), miss_precise_down(i, j));
        }
    }

}
