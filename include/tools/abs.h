#ifndef ABS_H
#define ABS_H

#include <cuda_fp16.h>

#include <stdexcept>

namespace cascade::abs_ns {

    template <typename TPrecision>
    TPrecision abs(const TPrecision &val) {
        throw std::runtime_error(
            "abs: reached umimplimented default function"
        );
    }

    template <> double abs(const double &);
    template <> float abs(const float &);
    template <> __half abs(const __half &);

}

#endif