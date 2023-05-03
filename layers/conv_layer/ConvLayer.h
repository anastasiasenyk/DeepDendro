//
// Created by Yaroslav Korch on 18.04.2023.
//

#ifndef DEEPDENDRO_CONVLAYER_H
#define DEEPDENDRO_CONVLAYER_H


#include "activationFuncs.h"
#include "activationDerivative.h"
#include <unsupported/Eigen/CXX11/Tensor>
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
    ConvLT a_values;

    Shape one_convolved_shape;
    Shape one_pooled_shape;

    ConvLT convolved_output;
    ConvLT pooled_output;

    MaxPool<ConvLDimension> default_pool;


public:

    explicit ConvLayer(Shape filter_shape, Shape input_shape, PoolParameters pool_parameters = {PoolType::MAX, 2, 2});

    auto convolve(const Filter<ConvLDimension> &filter);

    void convolve_all();

    void print() {
        std::cout << convolved_output << std::endl;
    }


};

template<size_t N_Filters, size_t ConvLDimension>
ConvLayer<N_Filters, ConvLDimension>::ConvLayer(const Shape filter_shape, const Shape input_shape,
                                                const PoolParameters pool_args): filter_shape{
        filter_shape}, filters{}, a_values{}, convolved_output{}, default_pool{} {

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

    a_values.resize(input_shape);

    // TODO: remove random initialization
    a_values.setRandom();
}

template<size_t N_Filters, size_t ConvLDimension>
auto ConvLayer<N_Filters, ConvLDimension>::convolve(const Filter<ConvLDimension> &filter) {
    return filter.convolve(a_values);
}

template<size_t N_Filters, size_t ConvLDimension>
void ConvLayer<N_Filters, ConvLDimension>::convolve_all() {

    Eigen::Tensor<double, ConvLDimension> conv_res;
    Eigen::Tensor<double, ConvLDimension> pool_res;

    auto to_combine_conv_start = one_convolved_shape;
    auto to_combine_pool_start = one_pooled_shape;

    auto dim_increment = one_convolved_shape[ConvLDimension - 1];

    for (int i = 0; i < ConvLDimension; ++i) {
        to_combine_conv_start[i] = 0;
        to_combine_pool_start[i] = 0;
    }

    for (size_t i = 0; i < N_Filters; ++i) {
        conv_res = convolve(filters[i]);

        convolved_output.slice(to_combine_conv_start, one_convolved_shape) = conv_res;

        default_pool.pool3D(conv_res);
        pooled_output.slice(to_combine_pool_start, one_pooled_shape) = default_pool.get_output();

        to_combine_conv_start[ConvLDimension - 1] += dim_increment;
        to_combine_pool_start[ConvLDimension - 1] += dim_increment;
    }

}


#endif //DEEPDENDRO_CONVLAYER_H
