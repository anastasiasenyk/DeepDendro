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
    ConvLT convolved_output;


public:

    explicit ConvLayer(Shape filter_shape, Shape input_shape);

    auto convolve(const Filter<ConvLDimension> &filter);

    void convolve_all();

    void print() {
        std::cout << convolved_output << std::endl;
    }


};

template<size_t N_Filters, size_t ConvLDimension>
ConvLayer<N_Filters, ConvLDimension>::ConvLayer(const Shape filter_shape, const Shape input_shape): filter_shape{
        filter_shape}, filters{}, a_values{}, convolved_output{} {

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

        convolved_shape[ConvLDimension - 1] *= N_Filters;

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

    auto to_combine_start = one_convolved_shape;
    auto dim_increment = one_convolved_shape[ConvLDimension - 1];

    for (int i = 0; i < ConvLDimension; ++i) {
        to_combine_start[i] = 0;
    }

    MaxPool<ConvLDimension> default_pool = DefaultPool<ConvLDimension>(2, 2, one_convolved_shape);

    for (size_t i = 0; i < N_Filters; ++i) {
        conv_res = convolve(filters[i]);

        default_pool.pool3D(conv_res);


        convolved_output.slice(to_combine_start, one_convolved_shape) = conv_res;

        to_combine_start[ConvLDimension - 1] += dim_increment;
    }
}


#endif //DEEPDENDRO_CONVLAYER_H
