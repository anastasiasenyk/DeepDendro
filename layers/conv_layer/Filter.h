//
// Created by Yaroslav Korch on 18.04.2023.
//

#ifndef DEEPDENDRO_FILTER_H
#define DEEPDENDRO_FILTER_H

#include "Layer.h"
#include "activationDerivative.h"
#include <unsupported/Eigen/CXX11/Tensor>

#include "activ_func_conv.h"
#include "common_funcs.h"
#include "Pooling.h"

template<size_t KernelDimension>
using ActivFunc = Eigen::Tensor<double, KernelDimension> (*)(Eigen::Tensor<double, KernelDimension> &);

template<size_t KernelDimension>
class Filter {
    using KernelT = Eigen::Tensor<double, KernelDimension>;
//    using PropertyArr = Eigen::array<size_t, KernelDimension>;
    using Shape = Eigen::array<Eigen::Index, KernelDimension>;

    Shape filter_shape;
    KernelT kernel_weights{};
    double bias{};

//    const PropertyArr padding;
//    const PropertyArr stride;
    Eigen::array<ptrdiff_t, KernelDimension> dims_to_convolve;

    ActivFunc<KernelDimension> activation_func;

    KernelT gradient_weights{};
    Eigen::Tensor<double, 0> gradient_bias{};

public:

    Filter() = default;

    explicit Filter(Shape filter_shape,
                    activation activation_func);

    auto get_weights() const {
        return kernel_weights;
    }

    KernelT convolve(const KernelT &input) const {
        KernelT res = input.convolve(kernel_weights, dims_to_convolve) + bias;
        return activation_func(res);
    }
};


template<size_t KernelDimension>
Filter<KernelDimension>::Filter(Shape filter_shape,
                                activation activation_func) :
        filter_shape(filter_shape) {

    {
        check_correct(no_zeros(filter_shape));
    }

    for (int i = 0; i < KernelDimension; ++i) {
        dims_to_convolve[i] = i;
    }

    // TODO: change for all activation functions
    this->activation_func = Tensor_ReLU<KernelDimension>;

    kernel_weights.resize(filter_shape);
    kernel_weights.setRandom();
}


#endif //DEEPDENDRO_FILTER_H
