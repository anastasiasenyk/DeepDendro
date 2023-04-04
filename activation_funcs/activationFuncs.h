//
// Created by Yaroslav Korch on 30.03.2023.
//

#ifndef DEEPDENDRO_ACTIVATIONFUNCS_H
#define DEEPDENDRO_ACTIVATIONFUNCS_H

#include <exception>
#include "Layer.h"

typedef MatrixXd (*ActivationFunc)(const MatrixXd &);

enum activation {
    sigmoid,
    relu,
    tanhyper,
    softmax
};

MatrixXd ReLU(const MatrixXd &input);

MatrixXd Sigmoid(const MatrixXd &input);

MatrixXd Tanh(const MatrixXd &input);

MatrixXd Softmax(const MatrixXd &input);

ActivationFunc find_activation_func(activation type);

class ActivationNotFound : public std::exception {
public:
    const char *what() const noexcept override {
        return "Unknown activation function";
    }
};

#endif //DEEPDENDRO_ACTIVATIONFUNCS_H
