//
// Created by Matthew Prytula on 04.04.2023.
//

#ifndef DEEPDENDRO_ACTIVATIONDERIVATIVE_H
#define DEEPDENDRO_ACTIVATIONDERIVATIVE_H

#include <exception>
#include "Layer.h"
#include "activationFuncs.h"
#include <unsupported/Eigen/CXX11/Tensor>


template<typename T>
T ReLUDer(const T &input) {
    return (input.array() > 0.0).template cast<double>();
}


template<typename T>
T SigmoidDer(const T &input) {
    // Compute the sigmoid activation function
    T sigmoid_x = 1.0 / (1.0 + (-input).array().exp());

    // Compute the derivative of the sigmoid activation function
    T sigmoid_deriv_x = sigmoid_x.array() * (1.0 - sigmoid_x.array());

    return sigmoid_deriv_x;
}

template<typename T>
T TanhDer(const MatrixXd &input) {
    // TODO: implement
    return input;
}

template<typename T, typename U>
T SoftmaxDer(const T &input) {
    T softmax = Softmax<T, U>(input);
    T diag_softmax = softmax.array() * (1.0 - softmax.array());
    return diag_softmax;
}



//class ActivationDerivativeNotFound : public std::exception {
//public:
//    [[nodiscard]] const char *what() const noexcept override {
//        return "Unknown  activation function. Can't find derivative";
//    }
//};

enum activation {
    sigmoid,
    relu,
    tanhyper,
    softmax
};

template<typename T>
struct ActivFuncs {
    ActivationFunc<T> activation_func;
    ActivationFunc<T> activation_func_der;
};

inline ActivFuncs<MatrixXd> find_activation_func_DENSE(activation type) {
    switch (type) {
        case relu:
            return {ReLU<MatrixXd>, ReLUDer<MatrixXd>};
        case sigmoid:
            return {Sigmoid<MatrixXd>, SigmoidDer<MatrixXd>};
        case tanhyper:
            throw ActivationNotFound();
            return {Tanh<MatrixXd>, TanhDer<MatrixXd>};
        case softmax:
            return {Softmax<MatrixXd, VectorXd>, SoftmaxDer<MatrixXd, VectorXd>};
        default:
            throw ActivationNotFound();
    }
}


template<size_t KernelDimension>
ActivationFunc<Eigen::Tensor<double, KernelDimension>> find_activation_func_FILTER(activation type) {
    using KernelT = Eigen::Tensor<double, KernelDimension>;

    switch (type) {
        case relu:
            return ReLU<KernelT>;
        case sigmoid:
            return Sigmoid<KernelT>;
        case tanhyper:
            throw ActivationNotFound();
//            return {Tanh<KernelT>, TanhDer<KernelT>};
        case softmax:
            // TODO: check if types are correct (in prev implementation it was VectorXd)
            return Softmax<KernelT, KernelT>;
        default:
            throw ActivationNotFound();
    }
}

#endif //DEEPDENDRO_ACTIVATIONDERIVATIVE_H
