#ifndef MATH_FUNCTIONS_H
#define MATH_FUNCTIONS_H

#include <cmath>
#include <cuda_fp16.h>

double square_root(double x);
float square_root(float x);
__half square_root(__half x);

#endif