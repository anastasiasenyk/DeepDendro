//
// Created by Yaroslav Korch on 30.03.2023.
//

#include "activationFuncs.h"


MatrixXd ReLU(const MatrixXd& input){
    return input.cwiseMax(0);
}

MatrixXd Sigmoid(const MatrixXd& input) {
    return 1.0 / (1.0 + (-input.array()).exp());
}

MatrixXd Tanh(const MatrixXd& input) {
    return input.array().tanh();
}

MatrixXd Softmax(const MatrixXd& input) {
    // applies softmax to every column of the matrix
    MatrixXd expMatrix = input.array().exp();
    VectorXd sumExp = expMatrix.colwise().sum();
    MatrixXd result = expMatrix.array().rowwise() / sumExp.transpose().array();
    return result;
}

ActivationFunc find_activation_func(activation type){
    switch (type){
        case relu:
            return ReLU;
        case sigmoid:
            return Sigmoid;
        case tanhyper:
            return Tanh;
        case softmax:
            return Softmax;
        default:
            throw ActivationNotFound();
    }
}
