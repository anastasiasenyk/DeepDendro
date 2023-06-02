//
// Created by Yaroslav Korch on 05.05.2023.
//

#ifndef DEEPDENDRO_FILTER_H
#define DEEPDENDRO_FILTER_H

#include <unsupported/Eigen/CXX11/Tensor>

#include "activationDerivative.h"
#include "activ_func_conv.h"
#include "common_funcs.h"
#include "Pooling.h"

template<size_t KernelDimension>
using ActivFunc = Eigen::Tensor<double, KernelDimension> (*)(const Eigen::Tensor<double, KernelDimension> &);

template<size_t KernelDimension>
class Filter {
    using KernelT = Eigen::Tensor<double, KernelDimension>;
    using Shape = Eigen::array<Eigen::Index, KernelDimension>;
    Shape filter_shape;
    KernelT kernel_weights{};
    double bias{};

    Eigen::array<ptrdiff_t, KernelDimension> dims_to_convolve;

    ActivFunc<KernelDimension> activation_func;
    ActivFunc<KernelDimension> activation_func_derivative;

    Eigen::array<int, KernelDimension> flip_order;

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

    void update_weights(const KernelT &dK, double dB, double lr) {
        kernel_weights -= lr * dK;
        bias -= lr * dB;
    }

    KernelT rotate_filter() const;

    KernelT activation_derivative(const KernelT &activated) const {
        return activation_func_derivative(activated);
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


    for (int i = 0; i < KernelDimension; i++) {
        flip_order[i] = 1;
    }


    // TODO: change for all activation functions
    this->activation_func = (activation_func == relu) ? Tensor_ReLU<KernelDimension> : Tensor_None<KernelDimension>;
    this->activation_func_derivative = (activation_func == relu) ? Tensor_ReLU_Derivative<KernelDimension>
                                                                 : Tensor_None_Derivative<KernelDimension>;

    kernel_weights.resize(filter_shape);
    kernel_weights.setRandom();
    kernel_weights = kernel_weights - 0.4;
}

template<size_t KernelDimension>
Eigen::Tensor<double, KernelDimension> Filter<KernelDimension>::rotate_filter() const {
    return kernel_weights.reverse(flip_order);
}

#endif //DEEPDENDRO_FILTER_H
