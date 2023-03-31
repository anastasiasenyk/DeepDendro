//
// Created by Yaroslav Korch on 30.03.2023.
//

#ifndef DEEPDENDRO_ACTIOVATION_FUNCS_H
#define DEEPDENDRO_ACTIOVATION_FUNCS_H

#include "Layer.h"

typedef MatrixXd (*ActivationFunc)(const MatrixXd&);

enum activation {
    sigmoid,
    relu,
    tanh,
    softmax
};

MatrixXd ReLU(const MatrixXd& input);

ActivationFunc find_activation_func(activation type);

class ActivationNotFound: public std::exception {
public:
    const char* what() const noexcept override {
        return "Unknown activation function";
    }
};

#endif //DEEPDENDRO_ACTIOVATION_FUNCS_H
