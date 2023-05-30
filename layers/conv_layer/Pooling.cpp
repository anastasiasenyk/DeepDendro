//
// Created by Yaroslav Korch on 06.05.2023.
//

#include "Pooling.h"

void MaxPool3D::pool3D(const KernelT &input) {
    double max_val, val;

    for (Eigen::Index i = 0; i < this->output_shape[0]; ++i) {
        for (Eigen::Index j = 0; j < this->output_shape[1]; ++j) {
            for (Eigen::Index k = 0; k < this->output_shape[2]; ++k) {
                Eigen::DSizes<Eigen::Index, 3> start(i * this->stride[0], j * this->stride[1], k * this->stride[2]);

                max_val = -std::numeric_limits<double>::infinity();
                for (Eigen::Index di = 0; di < this->grid_size[0]; ++di) {
                    for (Eigen::Index dj = 0; dj < this->grid_size[1]; ++dj) {
                        for (Eigen::Index dk = 0; dk < this->grid_size[2]; ++dk) {
                            val = input(start[0] + di, start[1] + dj, start[2] + dk);
                            max_val = std::max(max_val, val);
                        }
                    }
                }
                this->output(i, j, k) = max_val;
            }
        }
    }
}

MaxPool3D::KernelT
MaxPool3D::calc_back_prop(const KernelT &before_pool, const KernelT &after_pool, const KernelT &delta_piece) {
    check_correct(this->output_shape.at(0) == after_pool.dimension(0) &&
                  this->output_shape.at(1) == after_pool.dimension(1) &&
                  this->output_shape.at(2) == after_pool.dimension(2));

    dC.setZero();

    double max_val;
    for (Eigen::Index i = 0; i < this->output_shape[0]; ++i) {
        for (Eigen::Index j = 0; j < this->output_shape[1]; ++j) {
            for (Eigen::Index k = 0; k < this->output_shape[2]; ++k) {
                Eigen::DSizes<Eigen::Index, 3> start(i * this->stride[0], j * this->stride[1], k * this->stride[2]);
                max_val = after_pool(i, j, k);
                for (Eigen::Index di = 0; di < this->grid_size[0]; ++di) {
                    for (Eigen::Index dj = 0; dj < this->grid_size[1]; ++dj) {
                        for (Eigen::Index dk = 0; dk < this->grid_size[2]; ++dk) {
                            Eigen::DSizes<Eigen::Index, 3> idx(start[0] + di, start[1] + dj, start[2] + dk);
                            if (before_pool(idx[0], idx[1], idx[2]) == max_val) {
                                dC(idx[0], idx[1], idx[2]) = delta_piece(i, j, k);
                            }
                        }
                    }
                }
            }
        }
    }
    return dC;
}


void MaxPool2D::pool2D(const KernelT &input) {
    double max_val, val;

    for (Eigen::Index i = 0; i < this->output_shape[0]; ++i) {
        for (Eigen::Index j = 0; j < this->output_shape[1]; ++j) {
            Eigen::DSizes<Eigen::Index, 2> start(i * this->stride[0], j * this->stride[1]);

            max_val = -std::numeric_limits<double>::infinity();
            for (Eigen::Index di = 0; di < this->grid_size[0]; ++di) {
                for (Eigen::Index dj = 0; dj < this->grid_size[1]; ++dj) {
                    val = input(start[0] + di, start[1] + dj);
                    max_val = std::max(max_val, val);
                }
            }
            this->output(i, j) = max_val;
        }
    }
}

MaxPool2D::KernelT
MaxPool2D::calc_back_prop(const KernelT &before_pool, const KernelT &after_pool, const KernelT &delta_piece) {
    check_correct(this->output_shape.at(0) == after_pool.dimension(0) &&
                  this->output_shape.at(1) == after_pool.dimension(1));

    dC.setZero();

    double max_val;
    for (Eigen::Index i = 0; i < this->output_shape[0]; ++i) {
        for (Eigen::Index j = 0; j < this->output_shape[1]; ++j) {
            Eigen::DSizes<Eigen::Index, 2> start(i * this->stride[0], j * this->stride[1]);
            max_val = after_pool(i, j);
            for (Eigen::Index di = 0; di < this->grid_size[0]; ++di) {
                for (Eigen::Index dj = 0; dj < this->grid_size[1]; ++dj) {
                    Eigen::DSizes<Eigen::Index, 2> idx(start[0] + di, start[1] + dj);
                    if (before_pool(idx[0], idx[1]) == max_val) {
                        dC(idx[0], idx[1]) = delta_piece(i, j);
                    }
                }
            }
        }
    }
    return dC;
}
