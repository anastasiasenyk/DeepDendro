//
// Created by Yaroslav Korch on 05.05.2023.
//

#ifndef DEEPDENDRO_POOLING_H
#define DEEPDENDRO_POOLING_H

#include "cstddef"
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "common_funcs.h"

enum PoolType {
    MAX,
    AVG
};

template<size_t TensorDimension>
class MaxPoolingBase {
protected:
    using KernelT = Eigen::Tensor<double, TensorDimension>;
    using Shape = Eigen::array<Eigen::Index, TensorDimension>;

    Shape grid_size;
    Shape stride;
    KernelT output;
    KernelT dC;
public:
    Shape output_shape;

    MaxPoolingBase() = default;

    MaxPoolingBase(Shape input_shape, Shape grid_size, Shape stride);

    KernelT &get_output() {
        return output;
    }
};

template<size_t TensorDimension>
MaxPoolingBase<TensorDimension>::MaxPoolingBase(Shape input_shape, Shape grid_size, Shape stride) : grid_size(
        grid_size),
                                                                                                    stride(stride),
                                                                                                    output{} {
    {
        check_correct(no_zeros(grid_size));
        check_correct(no_zeros(stride));
        check_correct(no_zeros(input_shape));

        for (int i = 0; i < TensorDimension; ++i) {
            if (input_shape.at(i) < grid_size.at(i) || input_shape.at(i) < stride.at(i)) {
                throw std::invalid_argument("Input shape must be greater than grid size and stride");
            }
        }
    }

    for (size_t i = 0; i < TensorDimension; ++i) {
        output_shape[i] =
                std::ceil(static_cast<double>(input_shape[i] - grid_size[i]) / static_cast<double>(stride[i])) + 1;
    }
    dC.resize(input_shape);
    output.resize(output_shape);
}

class MaxPool3D : public MaxPoolingBase<3> {
    const static size_t TensorDimension = 3;
    using KernelT = Eigen::Tensor<double, TensorDimension>;
    using Shape = Eigen::array<Eigen::Index, TensorDimension>;

    void pool3D(const KernelT &input);

public:
    MaxPool3D(const Shape input_dims, const Shape grid_size,
              const Shape stride
    )
            : MaxPoolingBase<TensorDimension>(input_dims, grid_size, stride) {}

    KernelT forward_prop(const KernelT &input) {
        pool3D(input);
        return output;
    };

    KernelT calc_back_prop(const KernelT &before_pool, const KernelT &after_pool, const KernelT &delta_piece);
};


class MaxPool2D : public MaxPoolingBase<2> {
    const static size_t TensorDimension = 2;
    using KernelT = Eigen::Tensor<double, TensorDimension>;
    using Shape = Eigen::array<Eigen::Index, TensorDimension>;

    void pool2D(const KernelT &input);

public:
    MaxPool2D(const Shape input_dims, const Shape grid_size,
              const Shape stride
    )
            : MaxPoolingBase<TensorDimension>(input_dims, grid_size, stride) {}

    KernelT forward_prop(const KernelT &input) {
        pool2D(input);
        return output;
    };

    KernelT calc_back_prop(const KernelT &before_pool, const KernelT &after_pool, const KernelT &delta_piece);
};



#endif //DEEPDENDRO_POOLING_H
