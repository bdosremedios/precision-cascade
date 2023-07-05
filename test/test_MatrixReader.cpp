#include "gtest/gtest.h"
#include "Eigen/Dense"
#include "read_matrix/MatrixReader.h"
#include <string>

using mxread::MatrixReader;
using Eigen::MatrixXd;
using std::string;
using std::cout, std::endl;
using std::runtime_error;

class MatrixReaderTest: public testing::Test {

    protected:
        MatrixReader mr;
        string matrix_dir = "/home/bdosremedios/learn/gmres/test/read_matrices/";

    
    MatrixReaderTest() {
        mr = MatrixReader();
    }

};

TEST_F(MatrixReaderTest, ReadSquareMatrix) {

    MatrixXd target1 {{1, 2, 3},
                      {4, 5, 6},
                      {7, 8, 9}};
    MatrixXd target2 {{1, 2, 3, 4, 5},
                      {6, 7, 8, 9, 10},
                      {11, 12, 13, 14, 15},
                      {16, 17, 18, 19, 20},
                      {21, 22, 23, 24, 25}};
    string square1_file = matrix_dir + "square1.csv";
    string square2_file = matrix_dir + "square2.csv";
    MatrixXd test1(mr.read_file_d(square1_file));
    MatrixXd test2(mr.read_file_d(square2_file));

    // Check that read is correct for first file
    ASSERT_EQ(test1.rows(), 3);
    ASSERT_EQ(test1.cols(), 3);
    for (int i=0; i<3; ++i) {
        for (int j=0; j<3; ++j) {
            EXPECT_DOUBLE_EQ(test1(i, j), target1(i, j));
        }
    }

    // Check that read is correct for second file
    ASSERT_EQ(test2.rows(), 5);
    ASSERT_EQ(test2.cols(), 5);
    for (int i=0; i<5; ++i) {
        for (int j=0; j<5; ++j) {
            EXPECT_DOUBLE_EQ(test2(i, j), target2(i, j));
        }
    }

}

TEST_F(MatrixReaderTest, ReadWideTallMatrix) {

    MatrixXd target_wide {{10, 9, 8, 7, 6},
                          {5, 4, 3, 2, 1}};
    MatrixXd target_tall {{1, 2},
                          {3, 4},
                          {5, 6},
                          {7, 8}};
    string wide_file = matrix_dir + "wide.csv";
    string tall_file = matrix_dir + "tall.csv";
    MatrixXd test_wide(mr.read_file_d(wide_file));
    MatrixXd test_tall(mr.read_file_d(tall_file));

    // Check that read is correct for first file
    ASSERT_EQ(test_wide.rows(), 2);
    ASSERT_EQ(test_wide.cols(), 5);
    for (int i=0; i<2; ++i) {
        for (int j=0; j<5; ++j) {
            EXPECT_DOUBLE_EQ(test_wide(i, j), target_wide(i, j));
        }
    }

    // Check that read is correct for second file
    ASSERT_EQ(test_tall.rows(), 4);
    ASSERT_EQ(test_tall.cols(), 2);
    for (int i=0; i<4; ++i) {
        for (int j=0; j<2; ++j) {
            EXPECT_DOUBLE_EQ(test_tall(i, j), target_tall(i, j));
        }
    }
    
}

TEST_F(MatrixReaderTest, ReadEmptyMatrix) {

    string empty_file = matrix_dir + "empty.csv";
    MatrixXd test_empty(mr.read_file_d(empty_file));
    ASSERT_EQ(test_empty.rows(), 0);
    ASSERT_EQ(test_empty.cols(), 0);

}


TEST_F(MatrixReaderTest, ReadPreciseMatrix) {

    MatrixXd target_precise {{2.71828182845905, 3.71828182845905},
                             {4.71828182845904, 5.71828182845904}};
    string precise_file = matrix_dir + "precise.csv";
    MatrixXd test_precise(mr.read_file_d(precise_file));

    ASSERT_EQ(test_precise.rows(), 2);
    ASSERT_EQ(test_precise.cols(), 2);
    for (int i=0; i<2; ++i) {
        for (int j=0; j<2; ++j) {
            EXPECT_DOUBLE_EQ(test_precise(i, j), target_precise(i, j));
        }
    }

}

TEST_F(MatrixReaderTest, ReadBadFiles) {

    // Try to load non-existent file
    string bad_file_0 = matrix_dir + "thisfile";
    try {
        MatrixXd test(mr.read_file_d(bad_file_0));
        FAIL();
    } catch (runtime_error e) {
        EXPECT_EQ(
            e.what(),
            "Failed to read: " + bad_file_0
        );
    }

    // Try to load file with too small row
    string bad_file_1 = matrix_dir + "bad1.csv";
    try {
        MatrixXd test(mr.read_file_d(bad_file_1));
        FAIL();
    } catch (runtime_error e) {
        EXPECT_EQ(
            e.what(),
            "Error in: " + bad_file_1 + "\n" + "Row 3 does not meet column size of 3"
        );
    }

    // Try to load file with too big rows
    string bad_file_2 = matrix_dir + "bad2.csv";
    try {
        MatrixXd test(mr.read_file_d(bad_file_2));
        FAIL();
    } catch (runtime_error e) {
        EXPECT_EQ(
            e.what(),
            "Error in: " + bad_file_2 + "\n" + "Row 2 exceeds column size of 3"
        );
    }

    // Try to load file with invalid character argument
    string bad_file_3 = matrix_dir + "bad3.csv";
    try {
        MatrixXd test(mr.read_file_d(bad_file_3));
        FAIL();
    } catch (runtime_error e) {
        EXPECT_EQ(
            e.what(),
            "Error in: " + bad_file_3 + "\n" + "Invalid argument in file, failed to convert to numeric"
        );
    }
    
}