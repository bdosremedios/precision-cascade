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

//     template <typename T>
//     void TestAddSubAssignment() {

//         Vector<T> vec_1(
//             *handle_ptr,
//             {static_cast<T>(-42.), static_cast<T>(0.), static_cast<T>(0.6)}
//         );
//         Vector<T> vec_2(
//             *handle_ptr,
//             {static_cast<T>(-38.), static_cast<T>(0.5), static_cast<T>(-0.6)}
//         );

//         Vector<T> vec_3(vec_1);
//         vec_3 += vec_2;
//         ASSERT_VECTOR_EQ(vec_3, vec_1+vec_2);
//         vec_3 += vec_3;
//         ASSERT_VECTOR_EQ(vec_3, (vec_1+vec_2)*static_cast<T>(2.));

//         Vector<T> vec_4(vec_1);
//         vec_4 -= vec_2;
//         ASSERT_VECTOR_EQ(vec_4, vec_1-vec_2);
//         vec_4 -= vec_4;
//         ASSERT_VECTOR_EQ(vec_4, (vec_1+vec_2)*static_cast<T>(0.));

//     }

//     template <typename T>
//     void TestDot() {

//         // Pre-calculated
//         Vector<T> vec_1_dot(
//             *handle_ptr,
//             {static_cast<T>(-4.), static_cast<T>(3.4), static_cast<T>(0.),
//              static_cast<T>(-2.1), static_cast<T>(1.8)}
//         );
//         Vector<T> vec_2_dot(
//             *handle_ptr,
//             {static_cast<T>(9.), static_cast<T>(10.), static_cast<T>(1.5),
//              static_cast<T>(-4.5), static_cast<T>(2.)}
//         );
//         ASSERT_NEAR(static_cast<double>(vec_1_dot.dot(vec_2_dot)),
//                     11.05,
//                     11.05*Tol<T>::gamma(5));

//         // Random
//         Vector<T> vec_1_dot_r(Vector<T>::Random(*handle_ptr, 10));
//         Vector<T> vec_2_dot_r(Vector<T>::Random(*handle_ptr, 10));
//         T acc = static_cast<T>(0.);
//         for (int i=0; i<10; ++i) { acc += vec_1_dot_r.get_elem(i)*vec_2_dot_r.get_elem(i); }
//         ASSERT_NEAR(static_cast<double>(vec_1_dot_r.dot(vec_2_dot_r)),
//                     static_cast<double>(acc),
//                     2.*std::abs(10.*Tol<T>::gamma(10)));

//     }

//     template <typename T>
//     void TestNorm() {

//         // Pre-calculated
//         Vector<T> vec_norm(
//             *handle_ptr, 
//             {static_cast<T>(-8.), static_cast<T>(0.8), static_cast<T>(-0.6),
//              static_cast<T>(4.), static_cast<T>(0.)}
//         );
//         ASSERT_NEAR(vec_norm.norm(),
//                     static_cast<T>(9.),
//                     static_cast<T>(9.)*static_cast<T>(Tol<T>::gamma(5)));

//         // Random
//         Vector<T> vec_norm_r(Vector<T>::Random(*handle_ptr, 10));
//         ASSERT_NEAR(vec_norm_r.norm(),
//                     std::sqrt(vec_norm_r.dot(vec_norm_r)),
//                     std::sqrt(vec_norm_r.dot(vec_norm_r))*Tol<T>::gamma(10));

//     }

//     template <typename T>
//     void TestBadVecVecOps() {

//         const int m(7);
//         Vector<T> vec(Vector<T>::Random(*handle_ptr, m));
//         Vector<T> vec_too_small(Vector<T>::Random(*handle_ptr, m-3));
//         Vector<T> vec_too_large(Vector<T>::Random(*handle_ptr, m+2));

//         CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { vec + vec_too_small; });
//         CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { vec + vec_too_large; });

//         CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { vec - vec_too_small; });
//         CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { vec - vec_too_large; });

//         CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() mutable { vec += vec_too_small; });
//         CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() mutable { vec += vec_too_large; });

//         CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() mutable { vec -= vec_too_small; });
//         CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() mutable { vec -= vec_too_large; });

//         CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { vec.dot(vec_too_small); });
//         CHECK_FUNC_HAS_RUNTIME_ERROR(print_errors, [=]() { vec.dot(vec_too_large); });

//     }

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
// TEST_F(Vector_Test, TestAddSubAssignment) {
//     TestAddSubAssignment<__half>(); TestAddSubAssignment<float>(); TestAddSubAssignment<double>();
// }

// TEST_F(Vector_Test, TestDot) { TestDot<__half>(); TestDot<float>(); TestDot<double>(); }

// TEST_F(Vector_Test, TestNorm) { TestNorm<__half>(); TestNorm<float>(); TestNorm<double>(); }

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

// class Vector_Sort_Test: public TestBase
// {
// public:

//     template <typename T>
//     void simple_sort_even() {

//         Vector<T> vec(
//             *handle_ptr,
//             {static_cast<T>(3.519), static_cast<T>(8.525), static_cast<T>(3.978), static_cast<T>(8.645),
//              static_cast<T>(2.798), static_cast<T>(1.477), static_cast<T>(7.021), static_cast<T>(5.689),
//              static_cast<T>(6.185), static_cast<T>(6.315)}
//         );

//         std::vector<int> target({5, 4, 0, 2, 7, 8, 9, 6, 1, 3});
//         std::vector<int> test(vec.sort_indices());

//         ASSERT_EQ(target, test);

//     }

//     template <typename T>
//     void simple_sort_odd() {

//         Vector<T> vec(
//             *handle_ptr,
//             {static_cast<T>(7.063), static_cast<T>(8.824), static_cast<T>(5.430), static_cast<T>(5.107),
//              static_cast<T>(5.478), static_cast<T>(8.819), static_cast<T>(7.995), static_cast<T>(9.787),
//              static_cast<T>(4.139), static_cast<T>(8.946), static_cast<T>(4.861), static_cast<T>(1.678),
//              static_cast<T>(9.176)}
//         );

//         std::vector<int> target({11, 8, 10, 3, 2, 4, 0, 6, 5, 1, 9, 12, 7});
//         std::vector<int> test(vec.sort_indices());
    
//         ASSERT_EQ(target, test);

//     }

//     template <typename T>
//     void simple_sort_duplicates() {

//         constexpr int n(7);
//         Vector<T> vec(
//             *handle_ptr,
//             {static_cast<T>(9.433), static_cast<T>(3.950), static_cast<T>(1.776), static_cast<T>(7.016),
//              static_cast<T>(1.409), static_cast<T>(1.776), static_cast<T>(7.016)}
//         );

//         std::vector<int> target({4, 2, 5, 1, 3, 6, 0});
//         std::vector<int> test(vec.sort_indices());

//         // Check that sorted element order is the same for all elements
//         for (int i=0; i<n; ++i) {
//             ASSERT_EQ(vec.get_elem(target[i]), vec.get_elem(test[i]));
//         }

//     }

//     template <typename T>
//     void sorted_already() {

//         Vector<T> vec(
//             *handle_ptr,
//             {static_cast<T>(1), static_cast<T>(2), static_cast<T>(3), static_cast<T>(4),
//              static_cast<T>(5), static_cast<T>(6), static_cast<T>(7), static_cast<T>(8),
//              static_cast<T>(9), static_cast<T>(10)}
//         );

//         std::vector<int> target({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
//         std::vector<int> test(vec.sort_indices());

//         ASSERT_EQ(target, test);

//     }

//     template <typename T>
//     void one_element() {

//         Vector<T> vec(
//             *handle_ptr,
//             {static_cast<T>(1)}
//         );

//         std::vector<int> target({0});
//         std::vector<int> test(vec.sort_indices());

//         ASSERT_EQ(target, test);

//     }

// };

// TEST_F(Vector_Sort_Test, TestSimpleSort_EvenNum) { 
//     simple_sort_even<__half>();
//     simple_sort_even<float>();
//     simple_sort_even<double>();
// }

// TEST_F(Vector_Sort_Test, TestSimpleSort_OddNum) {
//     simple_sort_odd<__half>();
//     simple_sort_odd<float>();
//     simple_sort_odd<double>();
// }

// TEST_F(Vector_Sort_Test, TestSimpleSort_Dupes) {
//     simple_sort_duplicates<__half>();
//     simple_sort_duplicates<float>();
//     simple_sort_duplicates<double>();
// }

// TEST_F(Vector_Sort_Test, TestAlreadySorted) {
//     sorted_already<__half>();
//     sorted_already<float>();
//     sorted_already<double>();
// }

// TEST_F(Vector_Sort_Test, TestOneElement) {
//     one_element<__half>();
//     one_element<float>();
//     one_element<double>();
// }