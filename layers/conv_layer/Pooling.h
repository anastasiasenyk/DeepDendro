//
// Created by Yaroslav Korch on 02.05.2023.
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
class Pooling {
protected:
    using KernelT = Eigen::Tensor<double, TensorDimension>;
    using Shape = Eigen::array<Eigen::Index, TensorDimension>;

    Shape grid_size;
    Shape stride;
    KernelT output;
    Shape output_shape;

public:

    Pooling() = default;

    Pooling(Shape grid_size, Shape stride, Shape input_shape);

    KernelT &get_output() {
        return output;
    }

    virtual void pool(const KernelT &input);
};

template<size_t TensorDimension>
void Pooling<TensorDimension>::pool(const Pooling::KernelT &input) {

}

template<size_t TensorDimension>
Pooling<TensorDimension>::Pooling(Shape grid_size, Shape stride, Shape input_shape) : grid_size(grid_size),
                                                                                      stride(stride), output{} {
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

    output.resize(output_shape);
}


template<size_t TensorDimension>
class MaxPool : public Pooling<TensorDimension> {
    using KernelT = Eigen::Tensor<double, TensorDimension>;
    using Shape = Eigen::array<Eigen::Index, TensorDimension>;

public:

    MaxPool(const Shape &grid_size,
            const Shape &stride,
            const Shape &input_dims)
            : Pooling<TensorDimension>(grid_size, stride, input_dims) {}


//    void pool(const KernelT &input) override;
    void pool3D(const Eigen::Tensor<double, 3> &input);
};

template<typename Scalar>
Scalar find_maximum(const Eigen::Tensor<Scalar, 3> &tensor) {
    Scalar max_val = tensor(0, 0, 0);
    for (Eigen::Index i = 0; i < tensor.dimension(0); ++i) {
        for (Eigen::Index j = 0; j < tensor.dimension(1); ++j) {
            for (Eigen::Index k = 0; k < tensor.dimension(2); ++k) {
                if (tensor(i, j, k) > max_val) {
                    max_val = tensor(i, j, k);
                }
            }
        }
    }
    return max_val;
}


template<size_t TensorDimension>
void MaxPool<TensorDimension>::pool3D(const Eigen::Tensor<double, 3>& input) {
    if (TensorDimension != 3) {
        throw std::invalid_argument("MaxPool is only implemented for 3D tensors");
    }

    for (Eigen::Index i = 0; i < this->output_shape[0]; ++i) {
        for (Eigen::Index j = 0; j < this->output_shape[1]; ++j) {
            for (Eigen::Index k = 0; k < this->output_shape[2]; ++k) {
                Eigen::DSizes<Eigen::Index, 3> start(i * this->stride[0], j * this->stride[1], k * this->stride[2]);

                double max_val = -std::numeric_limits<double>::infinity();
                for (Eigen::Index di = 0; di < this->grid_size[0]; ++di) {
                    for (Eigen::Index dj = 0; dj < this->grid_size[1]; ++dj) {
                        for (Eigen::Index dk = 0; dk < this->grid_size[2]; ++dk) {
                            double val = input(start[0] + di, start[1] + dj, start[2] + dk);
                            if (val > max_val) {
                                max_val = val;
                            }
                        }
                    }
                }
                this->output(i, j, k) = max_val;
            }
        }
    }
}


template<size_t TensorDimension>
class AvgPool : public Pooling<TensorDimension> {
    using KernelT = Eigen::Tensor<double, TensorDimension>;
    using Shape = Eigen::array<Eigen::Index, TensorDimension>;
public:
    AvgPool(const Shape &grid_size,
            const Shape &stride,
            const Shape &input_dims)
            : Pooling<TensorDimension>(grid_size, stride, input_dims) {}

};


template<size_t TensorDimension>
MaxPool<TensorDimension> DefaultPool(const size_t f, const size_t s,
                                     Eigen::array<Eigen::Index, TensorDimension> input_dims) {
    using Shape = Eigen::array<Eigen::Index, TensorDimension>;
    Shape grid, stride;

    for (size_t i = 0; i < TensorDimension; ++i) {
        grid[i] = f;
        stride[i] = s;
    }

    grid[TensorDimension - 1] = 1;
    stride[TensorDimension - 1] = 1;

    return MaxPool<TensorDimension>(grid, stride, input_dims);
}

#endif //DEEPDENDRO_POOLING_H
