#include "../test.h"

#include "types/Scalar.h"

class Scalar_Test: public TestBase
{
public:

    template <typename T>
    void TestGetSet() {

        Scalar<T> scalar_1;
        Scalar<T> scalar_2;

        scalar_1.set_scalar(static_cast<T>(5.));
        scalar_2.set_scalar(static_cast<T>(-3.1));

        ASSERT_EQ(scalar_1.get_scalar(), static_cast<T>(5.));
        ASSERT_EQ(scalar_2.get_scalar(), static_cast<T>(-3.1));

        scalar_1.set_scalar(static_cast<T>(0.));
        scalar_2.set_scalar(static_cast<T>(5.));

        ASSERT_EQ(scalar_1.get_scalar(), static_cast<T>(0.));
        ASSERT_EQ(scalar_2.get_scalar(), static_cast<T>(5.));

    }

    template <typename T>
    void TestConstruction() {

        Scalar<T> scalar_1;
        Scalar<T> scalar_2(static_cast<T>(5.));
        Scalar<T> scalar_3(static_cast<T>(-1.2));
        Scalar<T> scalar_4(static_cast<T>(0.));

        ASSERT_EQ(scalar_2.get_scalar(), static_cast<T>(5.));
        ASSERT_EQ(scalar_3.get_scalar(), static_cast<T>(-1.2));
        ASSERT_EQ(scalar_4.get_scalar(), static_cast<T>(0.));
    }

    template <typename T>
    void TestCopyAssignment() {

        Scalar<T> scalar_1(static_cast<T>(5.));
        scalar_1 = Scalar<T>(static_cast<T>(-1.2));
        ASSERT_EQ(scalar_1.get_scalar(), static_cast<T>(-1.2));
        scalar_1 = Scalar<T>(static_cast<T>(0.));
        ASSERT_EQ(scalar_1.get_scalar(), static_cast<T>(0.));

    }

    template <typename T>
    void TestCopyConstruction() {

        Scalar<T> scalar_1(static_cast<T>(5.));
        Scalar<T> scalar_2(scalar_1);
        ASSERT_EQ(scalar_2.get_scalar(), static_cast<T>(5.));

        Scalar<T> scalar_3(static_cast<T>(-42.5));
        Scalar<T> scalar_4(scalar_3);
        ASSERT_EQ(scalar_4.get_scalar(), static_cast<T>(-42.5));

    }

    template <typename T>
    void TestAddSub() {

        Scalar<T> scalar_1(static_cast<T>(3.2));
        Scalar<T> scalar_2(static_cast<T>(5.));
        Scalar<T> scalar_3(static_cast<T>(-4.));

        ASSERT_NEAR(
            (scalar_1+scalar_2).get_scalar(),
            static_cast<T>(8.2),
            min_1_mag(static_cast<T>(8.2))*Tol<T>::roundoff_T()
        );
        ASSERT_NEAR(
            (scalar_1+scalar_3).get_scalar(),
            static_cast<T>(-0.8),
            min_1_mag(static_cast<T>(-0.8))*Tol<T>::roundoff_T()
        );
        ASSERT_NEAR(
            (scalar_2+scalar_3).get_scalar(),
            static_cast<T>(1.),
            min_1_mag(static_cast<T>(1.))*Tol<T>::roundoff_T()
        );
        ASSERT_NEAR(
            (scalar_1+scalar_1).get_scalar(),
            static_cast<T>(6.4),
            min_1_mag(static_cast<T>(6.4))*Tol<T>::roundoff_T()
        );
        
        ASSERT_NEAR(
            (scalar_2+scalar_1).get_scalar(),
            static_cast<T>(8.2),
            min_1_mag(static_cast<T>(8.2))*Tol<T>::roundoff_T()
        );
        ASSERT_NEAR(
            (scalar_3+scalar_1).get_scalar(),
            static_cast<T>(-0.8),
            min_1_mag(static_cast<T>(-0.8))*Tol<T>::roundoff_T()
        );
        ASSERT_NEAR(
            (scalar_3+scalar_2).get_scalar(),
            static_cast<T>(1.),
            min_1_mag(static_cast<T>(1.))*Tol<T>::roundoff_T()
        );

        ASSERT_NEAR(
            (scalar_1-scalar_2).get_scalar(),
            static_cast<T>(-1.8),
            min_1_mag(static_cast<T>(-1.8))*Tol<T>::roundoff_T()
        );
        ASSERT_NEAR(
            (scalar_1-scalar_3).get_scalar(),
            static_cast<T>(7.2),
            min_1_mag(static_cast<T>(7.2))*Tol<T>::roundoff_T()
        );
        ASSERT_NEAR(
            (scalar_2-scalar_3).get_scalar(),
            static_cast<T>(9.),
            min_1_mag(static_cast<T>(9.))*Tol<T>::roundoff_T()
        );
        ASSERT_NEAR(
            (scalar_1-scalar_1).get_scalar(),
            static_cast<T>(0.),
            min_1_mag(static_cast<T>(0.))*Tol<T>::roundoff_T()
        );
        
        ASSERT_NEAR(
            (scalar_2-scalar_1).get_scalar(),
            static_cast<T>(1.8),
            min_1_mag(static_cast<T>(1.8))*Tol<T>::roundoff_T()
        );
        ASSERT_NEAR(
            (scalar_3-scalar_1).get_scalar(),
            static_cast<T>(-7.2),
            min_1_mag(static_cast<T>(-7.2))*Tol<T>::roundoff_T()
        );
        ASSERT_NEAR(
            (scalar_3-scalar_2).get_scalar(),
            static_cast<T>(-9.),
            min_1_mag(static_cast<T>(-9.))*Tol<T>::roundoff_T()
        );

    }
    
    template <typename T>
    void TestAddSubAssignment() {

        Scalar<T> scalar_1(static_cast<T>(3.2));
        Scalar<T> scalar_2(static_cast<T>(5.));
        Scalar<T> scalar_3(static_cast<T>(-4.));

        scalar_2 += scalar_1;
        ASSERT_NEAR(
            scalar_2.get_scalar(),
            static_cast<T>(8.2),
            min_1_mag(static_cast<T>(8.2))*Tol<T>::roundoff_T()
        );
        scalar_3 += scalar_1;
        ASSERT_NEAR(
            scalar_3.get_scalar(),
            static_cast<T>(-0.8),
            min_1_mag(static_cast<T>(-0.8))*Tol<T>::roundoff_T()
        );
        scalar_3 += scalar_1;
        ASSERT_NEAR(
            scalar_3.get_scalar(),
            static_cast<T>(2.4),
            min_1_mag(static_cast<T>(2.4))*Tol<T>::roundoff_T()
        );
        scalar_1 += scalar_1;
        ASSERT_NEAR(
            scalar_1.get_scalar(),
            static_cast<T>(6.4),
            min_1_mag(static_cast<T>(6.4))*Tol<T>::roundoff_T()
        );

        scalar_1.set_scalar(static_cast<T>(3.2));
        scalar_2.set_scalar(static_cast<T>(5.));
        scalar_3.set_scalar(static_cast<T>(-4.));
        scalar_2 -= scalar_1;
        ASSERT_NEAR(
            scalar_2.get_scalar(),
            static_cast<T>(1.8),
            min_1_mag(static_cast<T>(1.8))*Tol<T>::roundoff_T()
        );
        scalar_3 -= scalar_1;
        ASSERT_NEAR(
            scalar_3.get_scalar(),
            static_cast<T>(-7.2),
            min_1_mag(static_cast<T>(-7.2))*Tol<T>::roundoff_T()
        );
        scalar_3 -= scalar_1;
        ASSERT_NEAR(
            scalar_3.get_scalar(),
            static_cast<T>(-10.4),
            min_1_mag(static_cast<T>(-10.4))*Tol<T>::roundoff_T()
        );
        scalar_1 -= scalar_1;
        ASSERT_NEAR(
            scalar_1.get_scalar(),
            static_cast<T>(0.),
            min_1_mag(static_cast<T>(0.))*Tol<T>::roundoff_T()
        );

    }

    template <typename T>
    void TestMultDiv() {

        Scalar<T> scalar_1(static_cast<T>(-3.));
        Scalar<T> scalar_2(static_cast<T>(2.4));
        Scalar<T> scalar_3(static_cast<T>(5.));
        Scalar<T> scalar_4(static_cast<T>(0.));

        ASSERT_NEAR(
            (scalar_1*scalar_1).get_scalar(),
            static_cast<T>(9.),
            min_1_mag(static_cast<T>(9.))*Tol<T>::roundoff_T()
        );
        ASSERT_NEAR(
            (scalar_1*scalar_2).get_scalar(),
            static_cast<T>(-7.2),
            min_1_mag(static_cast<T>(-7.2))*Tol<T>::roundoff_T()
        );
        ASSERT_NEAR(
            (scalar_1*scalar_3).get_scalar(),
            static_cast<T>(-15.),
            min_1_mag(static_cast<T>(-15.))*Tol<T>::roundoff_T()
        );
        ASSERT_NEAR(
            (scalar_1*scalar_4).get_scalar(),
            static_cast<T>(0.),
            min_1_mag(static_cast<T>(0.))*Tol<T>::roundoff_T()
        );

        ASSERT_NEAR(
            (scalar_2*scalar_1).get_scalar(),
            static_cast<T>(-7.2),
            min_1_mag(static_cast<T>(-7.2))*Tol<T>::roundoff_T()
        );
        ASSERT_NEAR(
            (scalar_2*scalar_2).get_scalar(),
            static_cast<T>(5.76),
            min_1_mag(static_cast<T>(5.76))*Tol<T>::roundoff_T()
        );
        ASSERT_NEAR(
            (scalar_2*scalar_3).get_scalar(),
            static_cast<T>(12.),
            min_1_mag(static_cast<T>(12.))*Tol<T>::roundoff_T()
        );
        ASSERT_NEAR(
            (scalar_2*scalar_4).get_scalar(),
            static_cast<T>(0.),
            min_1_mag(static_cast<T>(0.))*Tol<T>::roundoff_T()
        );

        ASSERT_NEAR(
            (scalar_1/scalar_1).get_scalar(),
            static_cast<T>(1.),
            static_cast<T>(2)*min_1_mag(static_cast<T>(1.))*Tol<T>::roundoff_T()
        );
        ASSERT_NEAR(
            (scalar_1/scalar_2).get_scalar(),
            static_cast<T>(-1.25),
            static_cast<T>(2)*min_1_mag(static_cast<T>(-1.25))*Tol<T>::roundoff_T()
        );
        ASSERT_NEAR(
            (scalar_1/scalar_3).get_scalar(),
            static_cast<T>(-3./5.),
            static_cast<T>(2)*min_1_mag(static_cast<T>(-3./5.))*Tol<T>::roundoff_T()
        );

        ASSERT_NEAR(
            (scalar_2/scalar_1).get_scalar(),
            static_cast<T>(-0.8),
            static_cast<T>(2)*min_1_mag(static_cast<T>(-0.8))*Tol<T>::roundoff_T()
        );
        ASSERT_NEAR(
            (scalar_2/scalar_2).get_scalar(),
            static_cast<T>(1.),
            static_cast<T>(2)*min_1_mag(static_cast<T>(1.))*Tol<T>::roundoff_T()
        );
        ASSERT_NEAR(
            (scalar_2/scalar_3).get_scalar(),
            static_cast<T>(0.48),
            static_cast<T>(2)*min_1_mag(static_cast<T>(0.48))*Tol<T>::roundoff_T()
        );

    }

    template <typename T>
    void TestMultDivAssignment() {

        Scalar<T> scalar_1(static_cast<T>(-3.));
        Scalar<T> scalar_2(static_cast<T>(2.4));
        Scalar<T> scalar_3(static_cast<T>(5.));
        Scalar<T> scalar_4(static_cast<T>(0.));

        scalar_2 *= scalar_1;
        ASSERT_NEAR(
            scalar_2.get_scalar(),
            static_cast<T>(-7.2),
            min_1_mag(static_cast<T>(-7.2))*Tol<T>::roundoff_T()
        );
        scalar_3 *= scalar_1;
        ASSERT_NEAR(
            scalar_3.get_scalar(),
            static_cast<T>(-15.),
            min_1_mag(static_cast<T>(-15.))*Tol<T>::roundoff_T()
        );
        scalar_4 *= scalar_1;
        ASSERT_NEAR(
            scalar_4.get_scalar(),
            static_cast<T>(0.),
            min_1_mag(static_cast<T>(0.))*Tol<T>::roundoff_T()
        );
        scalar_1 *= scalar_1;
        ASSERT_NEAR(
            scalar_1.get_scalar(),
            static_cast<T>(9.),
            min_1_mag(static_cast<T>(9.))*Tol<T>::roundoff_T()
        );

        scalar_1.set_scalar(static_cast<T>(-3.));
        scalar_2.set_scalar(static_cast<T>(2.4));
        scalar_3.set_scalar(static_cast<T>(5.));

        scalar_2 /= scalar_1;
        ASSERT_NEAR(
            scalar_2.get_scalar(),
            static_cast<T>(-0.8),
            static_cast<T>(2)*min_1_mag(static_cast<T>(-0.8))*Tol<T>::roundoff_T()
        );
        scalar_3 /= scalar_1;
        ASSERT_NEAR(
            scalar_3.get_scalar(),
            static_cast<T>(5./-3.),
            static_cast<T>(2)*min_1_mag(static_cast<T>(5./-3.))*Tol<T>::roundoff_T()
        );
        scalar_1 /= scalar_1;
        ASSERT_NEAR(
            scalar_1.get_scalar(),
            static_cast<T>(1.),
            static_cast<T>(2)*min_1_mag(static_cast<T>(1.))*Tol<T>::roundoff_T()
        );

    }

    template <typename T>
    void TestBooleanEqual() {

        Scalar<T> scalar_1(static_cast<T>(7.1));
        Scalar<T> scalar_2(static_cast<T>(-7.1));
        Scalar<T> scalar_3(static_cast<T>(7.1));
        Scalar<T> scalar_4(static_cast<T>(3.2));

        ASSERT_TRUE(scalar_1 == scalar_1);
        ASSERT_TRUE(scalar_1 == scalar_3);
        ASSERT_TRUE(scalar_3 == scalar_1);

        ASSERT_FALSE(scalar_1 == scalar_2);
        ASSERT_FALSE(scalar_2 == scalar_1);
        ASSERT_FALSE(scalar_1 == scalar_4);
        ASSERT_FALSE(scalar_4 == scalar_1);

    }

    template <typename T>
    void TestAbs() {

        Scalar<T> scalar_1(static_cast<T>(4.2));
        Scalar<T> scalar_2(static_cast<T>(-5.6));
        Scalar<T> scalar_3(static_cast<T>(0.));
        Scalar<T> scalar_4(static_cast<T>(-100.));

        ASSERT_EQ(scalar_1.abs().get_scalar(), static_cast<T>(4.2));
        ASSERT_EQ(scalar_2.abs().get_scalar(), static_cast<T>(5.6));
        ASSERT_EQ(scalar_3.abs().get_scalar(), static_cast<T>(0.));
        ASSERT_EQ(scalar_4.abs().get_scalar(), static_cast<T>(100.));

        ASSERT_EQ(scalar_1.get_scalar(), static_cast<T>(4.2));
        ASSERT_EQ(scalar_2.get_scalar(), static_cast<T>(5.6));
        ASSERT_EQ(scalar_3.get_scalar(), static_cast<T>(0.));
        ASSERT_EQ(scalar_4.get_scalar(), static_cast<T>(100.));

    }

    template <typename T>
    void TestSqrt() {

        Scalar<T> scalar_1(static_cast<T>(9.));
        Scalar<T> scalar_2(static_cast<T>(0.64));
        Scalar<T> scalar_3(static_cast<T>(100.));

        ASSERT_NEAR(scalar_1.sqrt().get_scalar(), static_cast<T>(3.), Tol<T>::roundoff_T());
        ASSERT_NEAR(scalar_2.sqrt().get_scalar(), static_cast<T>(0.8), Tol<T>::roundoff_T());
        ASSERT_NEAR(scalar_3.sqrt().get_scalar(), static_cast<T>(10.), Tol<T>::roundoff_T());

        ASSERT_NEAR(scalar_1.get_scalar(), static_cast<T>(3.), Tol<T>::roundoff_T());
        ASSERT_NEAR(scalar_2.get_scalar(), static_cast<T>(0.8), Tol<T>::roundoff_T());
        ASSERT_NEAR(scalar_3.get_scalar(), static_cast<T>(10.), Tol<T>::roundoff_T());

    }

//     void TestCast() {
        
//         constexpr int m(20);
//         Vector<double> vec_dbl(Vector<double>::Random(*handle_ptr, m));

//         Vector<float> vec_sgl(vec_dbl.cast<float>());
//         ASSERT_EQ(vec_sgl.rows(), m);
//         for (int i=0; i<m; ++i) { ASSERT_EQ(vec_sgl.get_elem(i),
//                                             static_cast<float>(vec_dbl.get_elem(i))); }

//         Vector<__half> vec_hlf(vec_dbl.cast<__half>());
//         ASSERT_EQ(vec_hlf.rows(), m);
//         for (int i=0; i<m; ++i) { ASSERT_EQ(vec_hlf.get_elem(i),
//                                             static_cast<__half>(vec_dbl.get_elem(i))); }

//     }

//     template <typename T>
//     void TestBooleanEqual() {

//         Vector<T> vec_to_compare(
//             *handle_ptr,
//             {static_cast<T>(-1.5), static_cast<T>(4), static_cast<T>(0.),
//              static_cast<T>(101), static_cast<T>(-101), static_cast<T>(7)}
//         );
//         Vector<T> vec_same(
//             *handle_ptr,
//             {static_cast<T>(-1.5), static_cast<T>(4), static_cast<T>(0.),
//              static_cast<T>(101), static_cast<T>(-101), static_cast<T>(7)}
//         );
//         Vector<T> vec_diffbeg(
//             *handle_ptr,
//             {static_cast<T>(1.5), static_cast<T>(4), static_cast<T>(0.),
//              static_cast<T>(101), static_cast<T>(-101), static_cast<T>(7)}
//         );
//         Vector<T> vec_diffend(
//             *handle_ptr,
//             {static_cast<T>(-1.5), static_cast<T>(4), static_cast<T>(0.),
//              static_cast<T>(101), static_cast<T>(-101), static_cast<T>(70)}
//         );
//         Vector<T> vec_diffmid(
//             *handle_ptr,
//             {static_cast<T>(-1.5), static_cast<T>(4), static_cast<T>(-100.),
//              static_cast<T>(101), static_cast<T>(-101), static_cast<T>(7)}
//         );
//         Vector<T> vec_smaller(
//             *handle_ptr,
//             {static_cast<T>(-1.5), static_cast<T>(4), static_cast<T>(0.),
//              static_cast<T>(101)}
//         );
//         Vector<T> vec_bigger(
//             *handle_ptr,
//             {static_cast<T>(-1.5), static_cast<T>(4), static_cast<T>(0.),
//              static_cast<T>(101), static_cast<T>(-101), static_cast<T>(0)}
//         );
//         Vector<T> vec_empty_1(*handle_ptr, {});
//         Vector<T> vec_empty_2(*handle_ptr, {});

//         ASSERT_TRUE(vec_to_compare == vec_to_compare);
//         ASSERT_TRUE(vec_to_compare == vec_same);
//         ASSERT_FALSE(vec_to_compare == vec_diffbeg);
//         ASSERT_FALSE(vec_to_compare == vec_diffend);
//         ASSERT_FALSE(vec_to_compare == vec_diffmid);
//         ASSERT_FALSE(vec_to_compare == vec_smaller);
//         ASSERT_FALSE(vec_to_compare == vec_bigger);
//         ASSERT_TRUE(vec_empty_1 == vec_empty_2);

//     }

};

TEST_F(Scalar_Test, TestGetSet) {
    TestGetSet<__half>();
    TestGetSet<float>();
    TestGetSet<double>();
}

TEST_F(Scalar_Test, TestConstruction) {
    TestConstruction<__half>();
    TestConstruction<float>();
    TestConstruction<double>();
}

TEST_F(Scalar_Test, TestCopyAssignment) {
    TestCopyAssignment<__half>();
    TestCopyAssignment<float>();
    TestCopyAssignment<double>();
}

TEST_F(Scalar_Test, TestCopyConstruction) {
    TestCopyConstruction<__half>();
    TestCopyConstruction<float>();
    TestCopyConstruction<double>();
}

TEST_F(Scalar_Test, TestAddSub) {
    TestAddSub<__half>();
    TestAddSub<float>();
    TestAddSub<double>();
}

TEST_F(Scalar_Test, TestAddSubAssignment) {
    TestAddSubAssignment<__half>();
    TestAddSubAssignment<float>();
    TestAddSubAssignment<double>();
}

TEST_F(Scalar_Test, TestMultDiv) {
    TestMultDiv<__half>();
    TestMultDiv<float>();
    TestMultDiv<double>();
}

TEST_F(Scalar_Test, TestMultDivAssignment) {
    TestMultDivAssignment<__half>();
    TestMultDivAssignment<float>();
    TestMultDivAssignment<double>();
}

TEST_F(Scalar_Test, TestBooleanEqual) {
    TestBooleanEqual<__half>();
    TestBooleanEqual<float>();
    TestBooleanEqual<double>();
}

TEST_F(Scalar_Test, TestAbs) {
    TestAbs<__half>();
    TestAbs<float>();
    TestAbs<double>();
}

TEST_F(Scalar_Test, TestSqrt) {
    TestSqrt<__half>();
    TestSqrt<float>();
    TestSqrt<double>();
}

// TEST_F(Vector_Test, TestCast) { TestCast(); }

// TEST_F(Vector_Test, TestBadVecVecOps) {
//     TestBadVecVecOps<__half>();
//     TestBadVecVecOps<float>();
//     TestBadVecVecOps<double>();
// }

// TEST_F(Vector_Test, TestBooleanEqual) {
//     TestBooleanEqual<__half>();
//     TestBooleanEqual<float>();
//     TestBooleanEqual<double>();
// }