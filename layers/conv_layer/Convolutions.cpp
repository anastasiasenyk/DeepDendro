//
// Created by Yaroslav Korch on 06.05.2023.
//

#include "Convolutions.h"

Convolutional3D::Convolutional3D(const size_t n_filters, const Shape filters_shape, activation activ_func,
                                 const Shape input_shape) : ConvLayer<DIMENSION>(n_filters, filters_shape, activ_func,
                                                                                 input_shape) {}

Convolutional2D::Convolutional2D(const size_t n_filters, const Shape filters_shape, activation activ_func,
                                 const Shape input_shape) : ConvLayer<DIMENSION>(n_filters, filters_shape, activ_func,
                                                                                 input_shape) {}
