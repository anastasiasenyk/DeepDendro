//
// Created by Yaroslav Korch on 18.04.2023.
//

#ifndef DEEPDENDRO_CONVLAYER_H
#define DEEPDENDRO_CONVLAYER_H


#include "activationFuncs.h"
#include "activationDerivative.h"
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>
#include "common_funcs.h"
#include "Filter.h"
#include "Pooling.h"


template<size_t N_Filters, size_t ConvLDimension>
class ConvLayer {

    using ConvLT = Eigen::Tensor<double, ConvLDimension>;
    using Shape = Eigen::array<Eigen::Index, ConvLDimension>;
    using Filters = std::array<Filter<ConvLDimension>, N_Filters>;

    Filters filters;
    Shape filter_shape;
    ConvLT prev_a_values;

    Shape one_convolved_shape;
    Shape one_pooled_shape;

    ConvLT convolved_output;
    ConvLT pooled_output;

    MaxPool<ConvLDimension> default_pool;


public:

    explicit ConvLayer(Shape filter_shape, Shape input_shape, PoolParameters pool_parameters = {PoolType::MAX, 2, 2});

    auto convolve(const Filter<ConvLDimension> &filter);

    void forward_prop(const ConvLT &input);

    void calc_back_prop(const ConvLT &delta);

    void print() {
        auto line = [](const auto &s){std::cout << s << "\n";};
        line("Filter shape:");
        line(filter_shape.at(0));
        line(filter_shape.at(1));
        line(filter_shape.at(2));

        line("One convolved shape:");
        line(one_convolved_shape.at(0));
        line(one_convolved_shape.at(1));
        line(one_convolved_shape.at(2));

        line("One pooled shape:");
        line(one_pooled_shape.at(0));
        line(one_pooled_shape.at(1));
        line(one_pooled_shape.at(2));

        line("Convolved output:");
        line(convolved_output.dimensions());

        line("Pooled output:");
        line(pooled_output.dimensions());
    }
};

template<size_t N_Filters, size_t ConvLDimension>
ConvLayer<N_Filters, ConvLDimension>::ConvLayer(const Shape filter_shape, const Shape input_shape,
                                                const PoolParameters pool_args): filter_shape{
        filter_shape}, filters{}, prev_a_values{}, convolved_output{}, default_pool{} {

    {
        check_correct(no_zeros(filter_shape));
        check_correct(no_zeros(input_shape));
    }

    for (size_t i = 0; i < N_Filters; ++i) {
        filters[i] = Filter<ConvLDimension>(filter_shape, activation::relu);
    }

    {
        auto convolved_shape = input_shape;
        for (size_t i = 0; i < ConvLDimension; ++i) {
            convolved_shape[i] -= filter_shape[i] - 1;
        }

        one_convolved_shape = convolved_shape;
        default_pool = DefaultPool(pool_args.grid_size, pool_args.stride, one_convolved_shape);
        one_pooled_shape = default_pool.output_shape;

        convolved_shape[ConvLDimension - 1] *= N_Filters;
        auto pooled_shape = one_pooled_shape;
        pooled_shape[ConvLDimension - 1] *= N_Filters;

        pooled_output.resize(pooled_shape);
        convolved_output.resize(convolved_shape);
    }

    prev_a_values.resize(input_shape);
}

template<size_t N_Filters, size_t ConvLDimension>
auto ConvLayer<N_Filters, ConvLDimension>::convolve(const Filter<ConvLDimension> &filter) {
    return filter.convolve(prev_a_values);
}

template<size_t N_Filters, size_t ConvLDimension>
void ConvLayer<N_Filters, ConvLDimension>::forward_prop(const ConvLT &input) {
    prev_a_values = input;

    Eigen::Tensor<double, ConvLDimension> conv_res;
    Eigen::Tensor<double, ConvLDimension> pool_res;

    auto to_combine_start = one_convolved_shape;

    auto dim_increment = one_convolved_shape[ConvLDimension - 1];

    for (int i = 0; i < ConvLDimension; ++i) {
        to_combine_start[i] = 0;
    }

    for (size_t i = 0; i < N_Filters; ++i) {
        conv_res = convolve(filters[i]);

        convolved_output.slice(to_combine_start, one_convolved_shape) = conv_res;

        default_pool.pool3D(conv_res);
        pooled_output.slice(to_combine_start, one_pooled_shape) = default_pool.get_output();

        to_combine_start[ConvLDimension - 1] += dim_increment;
    }
}

template<size_t ConvLDimension>
Eigen::Tensor<double, ConvLDimension> rotate_filter(const Eigen::Tensor<double, ConvLDimension> &filter) {
    Eigen::array<int, 3> flip_order = {1, 2, 0};
    Eigen::Tensor<double, 3> rotated_filter = filter.reverse(flip_order);
    return rotated_filter;
}

template<size_t N_Filters, size_t ConvLDimension>
void ConvLayer<N_Filters, ConvLDimension>::calc_back_prop(const ConvLT &delta) {
    auto to_separate_start = one_convolved_shape;
    auto dim_increment = one_convolved_shape[ConvLDimension - 1];
    for (int i = 0; i < ConvLDimension; ++i) {
        to_separate_start[i] = 0;
    }
    const Eigen::array<ptrdiff_t, 3> dims_to_convolve({0, 1, 2});

    for (const auto &filter : filters){
        // the gradient marks are used as in scientific notations or amateur videos on YT

        ConvLT delta_piece = delta.slice(to_separate_start, one_pooled_shape);
        std::cout << "Delta part: " << delta_piece.dimensions() << "\n";

        ConvLT dC(one_convolved_shape);
        std::cout << "dC: " << dC.dimensions() << "\n";

        dC.setZero();
        auto before_pool = convolved_output.slice(to_separate_start, one_convolved_shape);
        auto after_pool = pooled_output.slice(to_separate_start, one_pooled_shape);
        default_pool.calc_grad_in_pool(before_pool, after_pool, dC, delta_piece);

        ConvLT dC_dZ = Tensor_ReLU_Derivative<3>(dC);
        std::cout << "dC_dZ: " << dC_dZ.dimensions() << "\n";

        ConvLT dZ = dC * dC_dZ;
        std::cout << "dZ: " << dZ.dimensions() << "\n";

        ConvLT dK = prev_a_values.convolve(dZ, dims_to_convolve);  // also, dF
        std::cout << "dK: " << dK.dimensions() << "\n";

        auto dB = dZ.sum();


        ConvLT rotated_filter_weights = rotate_filter<3>(filter.get_weights());
        // 2*pad - filter + 1 = 0 ~ same convolution => pad = (filter - 1) / 2
        Eigen::array<int, 3> padding_sizes;
        // TODO: understand why there is a coefficient 2 here
        for (int i = 0; i < ConvLDimension; ++i) {
            padding_sizes[i] = 2 * static_cast<int>((filter_shape[i] - 1) / 2);
        }

        Eigen::array<std::pair<Eigen::Index, Eigen::Index>, 3> paddings;
        paddings[0] = std::make_pair(padding_sizes[0], padding_sizes[0]);
        paddings[1] = std::make_pair(padding_sizes[1], padding_sizes[1]);
        paddings[2] = std::make_pair(padding_sizes[2], padding_sizes[2]);

        ConvLT dZ_padded = dZ.pad(paddings, 0);
        std::cout << "dZ_padded: " << dZ_padded.dimensions() << "\n";

        ConvLT dX = dZ_padded.convolve(rotated_filter_weights, dims_to_convolve);

        std::cout << "dX \n";
        std::cout << dX.dimensions() << "\n";

        to_separate_start[ConvLDimension - 1] += dim_increment;
    }
}

Eigen::MatrixXd flatten(const Eigen::Tensor<double, 3> &tensor) {
    Eigen::TensorMap<Eigen::Tensor<double, 1>> flattened_tensor(const_cast<double *>(tensor.data()), tensor.size());
    Eigen::Map<Eigen::MatrixXd> matrix(flattened_tensor.data(), tensor.dimension(0), tensor.dimension(1));
    return matrix;
}


#endif //DEEPDENDRO_CONVLAYER_H
