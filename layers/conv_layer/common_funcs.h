//
// Created by Yaroslav Korch on 02.05.2023.
//

#ifndef DEEPDENDRO_COMMON_FUNCS_H
#define DEEPDENDRO_COMMON_FUNCS_H


#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>

template<size_t Dimension>
void traverseTensor(const Eigen::Tensor<double, Dimension> &tensor, const std::function<void()> &func) {
    for (int i = 0; i < tensor.dimension(0); ++i) {
        const auto &subTensor = tensor.chip(i, 0);
        if (subTensor.dimensions() > 1) {
            traverseTensor(subTensor);
        } else {
            func(subTensor);
        }
    }
}

template<class Arr>
bool no_zeros(const Arr &arr) {
    for (const auto i: arr) {
        if (i == 0) return false;
    }
    return true;
}

inline void check_correct(bool exp){
    if (!exp){
        std::cerr << "Error! check definitions...";
        throw std::exception();
    }
}

inline Eigen::VectorXd flatten(const Eigen::Tensor<double, 3> &tensor) {
    Eigen::TensorMap<Eigen::Tensor<double, 1>> flattened_tensor(const_cast<double *>(tensor.data()), tensor.size());
    Eigen::Map<Eigen::VectorXd> vector(flattened_tensor.data(), tensor.dimension(0) * tensor.dimension(1) * tensor.dimension(2));
    return vector;
}





#endif //DEEPDENDRO_COMMON_FUNCS_H
