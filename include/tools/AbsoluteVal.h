#ifndef ABSOLUTEVAL_H
#define ABSOLUTEVAL_H

#include <stdexcept>
#include <cuda_fp16.h>

namespace abs_ns
{

    template <typename T>
    T abs(const T &val) {
        throw std::runtime_error("abs: reached umimplimented default function");
    }

    template <> double abs(const double &);
    template <> float abs(const float &);
    template <> __half abs(const __half &);

}

#endif