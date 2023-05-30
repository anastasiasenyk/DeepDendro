//
// Created by Yaroslav Korch on 30.03.2023.
//

#ifndef DEEPDENDRO_ACTIVATIONFUNCS_H
#define DEEPDENDRO_ACTIVATIONFUNCS_H

#include <exception>
#include "Layer.h"
#include <map>
#include <unsupported/Eigen/CXX11/Tensor>

template<typename T>
using ActivationFunc = T (*)(const T &input);



template<typename T>
T ReLU(const T &input) {
    return input.cwiseMax(0);
}

template<typename T>
T Sigmoid(const T &input) {
    return 1.0 / (1.0 + (-input.array()).exp());
}

template<typename T>
T Tanh(const T &input) {
    return input.array().tanh();
}

template<typename T, typename V>
T Softmax(const T &input) {
    // applies softmax to every column of the matrix
    T expMatrix = input.array().exp();
    V sumExp = expMatrix.colwise().sum();
    T result = expMatrix.array().rowwise() / sumExp.transpose().array();
    return result;
}


class ActivationNotFound : public std::exception {
public:
    [[nodiscard]] const char *what() const noexcept override {
        return "Unknown activation function";
    }
};


#endif //DEEPDENDRO_ACTIVATIONFUNCS_H
