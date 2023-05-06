//
// Created by Yaroslav Korch on 05.05.2023.
//

#ifndef DEEPDENDRO_CONVLAYERBASE_H
#define DEEPDENDRO_CONVLAYERBASE_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>
#include "Filter.h"
#include "Pooling.h"
#include "common_funcs.h"
#include <vector>


template<size_t ConvLDimension>
class ConvLayer {
protected:
    using ConvLT = Eigen::Tensor<double, ConvLDimension>;
    using Shape = Eigen::array<Eigen::Index, ConvLDimension>;
    using Filters = std::vector<Filter<ConvLDimension>>;

    const static size_t DIMENSION = ConvLDimension; // might be useful for derived classes

    Filters filters;
    Shape filter_shape;

    ConvLT prev_a_values;

    Shape one_convolved_shape;
    ConvLT convolved_output;
    Eigen::array<ptrdiff_t, ConvLDimension> dims_to_convolve;

    std::vector<ConvLT> dK_grads;
    std::vector<double> dB_grads;

    ConvLT convolve(const Filter<ConvLDimension> &filter);

public:
    ConvLayer(size_t n_filters, Shape filter_shape, activation activ_func, Shape input_shape);

    ConvLT& forward_prop(const ConvLT &input);

    ConvLT calc_back_prop(const ConvLT &delta);

    void apply_back_prop(double learning_rate);

    void print_structure();
};


template<size_t ConvLDimension>
Eigen::Tensor<double, ConvLDimension> ConvLayer<ConvLDimension>::convolve(const Filter<ConvLDimension> &filter) {
    return filter.convolve(prev_a_values);
}

template<size_t ConvLDimension>
ConvLayer<ConvLDimension>::ConvLayer(const size_t n_filters, const Shape filter_shape, activation activ_func,
                                     const Shape input_shape) {

    {
        check_correct(no_zeros(filter_shape));
        check_correct(no_zeros(input_shape));
    }

    this->filter_shape = filter_shape;

    filters.reserve(n_filters);
    dK_grads.reserve(n_filters);
    dB_grads.reserve(n_filters);

    prev_a_values.resize(input_shape);

    for (size_t i = 0; i < n_filters; ++i) {
        filters.emplace_back(Filter<ConvLDimension>(filter_shape, activ_func));
    }

    Shape convolved_shape = input_shape;
    for (size_t i = 0; i < ConvLDimension; ++i) {
        convolved_shape[i] -= filter_shape[i] - 1;
        dims_to_convolve[i] = i;
    }

    one_convolved_shape = convolved_shape;

    convolved_shape[ConvLDimension - 1] *= n_filters;
    convolved_output.resize(convolved_shape);
}

template<size_t ConvLDimension>
Eigen::Tensor<double, ConvLDimension>& ConvLayer<ConvLDimension>::forward_prop(const ConvLT &input) {
    prev_a_values = input;

    ConvLT conv_res;
    auto to_combine_start = one_convolved_shape;
    auto dim_increment = one_convolved_shape[ConvLDimension - 1];


    for (int i = 0; i < ConvLDimension; ++i) {
        to_combine_start[i] = 0;
    }

    for (const auto &filter: filters) {
        conv_res = convolve(filter);
        convolved_output.slice(to_combine_start, one_convolved_shape) = conv_res;
        to_combine_start[ConvLDimension - 1] += dim_increment;
    }
    return convolved_output;
}

template<size_t ConvLDimension>
Eigen::Tensor<double, ConvLDimension> ConvLayer<ConvLDimension>::calc_back_prop(const ConvLT &delta) {
    ConvLT dX;
    dX.resize(prev_a_values.dimensions());
    dX.setZero();
    Eigen::Tensor<double, 0> intermediate_sum;

    auto to_separate_start = one_convolved_shape;
    auto dim_increment = one_convolved_shape[ConvLDimension - 1];

    for (int i = 0; i < ConvLDimension; ++i) {
        to_separate_start[i] = 0;
    }

    Eigen::array<int, ConvLDimension> padding_sizes;
    Eigen::array<std::pair<Eigen::Index, Eigen::Index>, ConvLDimension> paddings;
    // TODO: understand why there is a coefficient 2 here
    // 2*pad - filter + 1 = 0 ~ same convolution => pad = (filter - 1) / 2
    for (int i = 0; i < ConvLDimension; ++i) {
        padding_sizes[i] = filter_shape[i] - 1;
        paddings[i] = std::make_pair(padding_sizes[i], padding_sizes[i]);
    }

    for (size_t i = 0; i < filters.size(); ++i) {
        // the gradient marks are used as in scientific notations or amateur videos on YT
        const Filter<ConvLDimension> &filter = filters[i];


        ConvLT delta_piece = delta.slice(to_separate_start, one_convolved_shape);

        ConvLT dO_dZ = filter.activation_derivative(delta_piece);

        // TODO: check the formula
        ConvLT dZ = delta_piece * dO_dZ;


        dK_grads[i] = prev_a_values.convolve(dZ, dims_to_convolve);
        intermediate_sum = dZ.sum();
        dB_grads[i] = intermediate_sum();


        ConvLT rotated_filter_weights = filter.rotate_filter();
        ConvLT dZ_padded = dZ.pad(paddings, 0);
        dX += dZ_padded.convolve(rotated_filter_weights, dims_to_convolve);
        to_separate_start[ConvLDimension - 1] += dim_increment;

#ifdef DEBUG
        std::cout << "Iteration " << i << "\n";
        std::cout << "Delta slice:\n " << delta_piece << "\n";
        std::cout << "dO_dZ:\n " << dO_dZ << "\n";
        std::cout << "dZ:\n " << dZ << "\n";
        std::cout << "dK_grads:\n" <<  dK_grads[i] << "\n";
        std::cout << "dB_grads:\n" << dB_grads[i] << "\n";
        std::cout << "dX: \n" << dX << "\n";
#endif

    }
    return dX * (1. / filters.size());
}


template<size_t ConvLDimension>
void ConvLayer<ConvLDimension>::apply_back_prop(const double learning_rate) {
    for (size_t i = 0; i < filters.size(); ++i) {
        filters[i].update_weights(dK_grads[i], dB_grads[i], learning_rate);
    }
}

template<size_t ConvLDimension>
void ConvLayer<ConvLDimension>::print_structure() {
    auto line = [](const auto &s) { std::cout << s << "\n"; };

    line("Number of filters: " + std::to_string(filters.size()));
    line("Filter shape:");
    for (const auto &dim: filter_shape) {
        line(dim);
    }

    line("One convolved shape:");
    for (const auto &dim: one_convolved_shape) {
        line(dim);
    }

    line("Convolved output:");
    line(convolved_output.dimensions());
}

#endif //DEEPDENDRO_CONVLAYERBASE_H
