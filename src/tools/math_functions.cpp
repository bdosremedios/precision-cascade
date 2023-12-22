#include "tools/math_functions.h"

double square_root(double x) { return std::sqrt(x); }

float square_root(float x) { return std::sqrt(x); }

__half square_root(__half x) { return static_cast<__half>(std::sqrt(static_cast<float>(x))); }