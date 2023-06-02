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

inline void printTensor3D(const Eigen::Tensor<double, 3>& tensor) {
    // Get tensor dimensions
    int dim1 = tensor.dimension(0);
    int dim2 = tensor.dimension(1);
    int dim3 = tensor.dimension(2);

    // Print tensor elements
    for (int i = 0; i < dim1; ++i) {
        for (int j = 0; j < dim2; ++j) {
            for (int k = 0; k < dim3; ++k) {
                std::cout << tensor(i, j, k) << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}



#endif //DEEPDENDRO_COMMON_FUNCS_H
