#include "test_tools.h"

// Half Constants
template<>
double Tol<__half>::roundoff() { return std::pow(2, -10); }
template<>
double Tol<__half>::stationary_conv_tol() { return 5*std::pow(10, -02); }
template<>
double Tol<__half>::krylov_conv_tol() { return 5*std::pow(10, -02); }
template<>
double Tol<__half>::nested_krylov_conv_tol() { return 5*std::pow(10, -02); }

// Single Constants
template<>
double Tol<float>::roundoff() { return std::pow(2, -23); }
template<>
double Tol<float>::stationary_conv_tol() { return 5*std::pow(10, -06); }
template<>
double Tol<float>::krylov_conv_tol() { return 5*std::pow(10, -06); }
template<>
double Tol<float>::nested_krylov_conv_tol() { return 5*std::pow(10, -06); }

// Double Constants
template<>
double Tol<double>::roundoff() { return std::pow(2, -52); }
template<>
double Tol<double>::stationary_conv_tol() { return std::pow(10, -10); }
template<>
double Tol<double>::krylov_conv_tol() { return std::pow(10, -10); }
template<>
double Tol<double>::nested_krylov_conv_tol() { return std::pow(10, -10); }