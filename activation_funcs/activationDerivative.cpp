//
// Created by Matthew Prytula on 04.04.2023.
//

#include "activationDerivative.h"

MatrixXd ReLUDer(const MatrixXd& input) {
    return (input.array() > 0.0).cast<double>();
}

MatrixXd SigmoidDer(const MatrixXd& input) {
    // Compute the sigmoid activation function
    MatrixXd sigmoid_x = 1.0 / (1.0 + (-input).array().exp());

    // Compute the derivative of the sigmoid activation function
    MatrixXd sigmoid_deriv_x = sigmoid_x.array() * (1.0 - sigmoid_x.array());

    return sigmoid_deriv_x;
}

MatrixXd TanhDer(const MatrixXd& input) {
    // TODO: implement
    return input;
}

MatrixXd SoftmaxDer(const MatrixXd& input) {
    // TODO: implement
    return input;
}


ActivationFunc find_activation_der(activation type){
    switch (type){
        case relu:
            return ReLUDer;
        case sigmoid:
            return SigmoidDer;
        case tanhyper:
            return TanhDer;
        case softmax:
            return SoftmaxDer;
        default:
            throw ActivationDerivativeNotFound();
    }
}