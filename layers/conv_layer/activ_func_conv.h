//
// Created by Yaroslav Korch on 02.05.2023.
//

#ifndef DEEPDENDRO_ACTIV_FUNC_CONV_H
#define DEEPDENDRO_ACTIV_FUNC_CONV_H

#include <unsupported/Eigen/CXX11/Tensor>


template<size_t Dimension>
Eigen::Tensor<double, Dimension> Tensor_ReLU(const Eigen::Tensor<double, Dimension> &tensor) {
    return tensor.unaryExpr([](double x) { return x > 0 ? x : 0; });
}


template<size_t Dimension>
Eigen::Tensor<double, Dimension> Tensor_ReLU_Derivative(const Eigen::Tensor<double, Dimension> &tensor) {
    return tensor.unaryExpr([](double x) { return static_cast<double>(x > 0); });
}

#endif //DEEPDENDRO_ACTIV_FUNC_CONV_H
