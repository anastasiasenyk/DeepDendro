//
// Created by Matthew Prytula on 04.04.2023.
//

#ifndef DEEPDENDRO_ACTIVATIONDERIVATIVE_H
#define DEEPDENDRO_ACTIVATIONDERIVATIVE_H

#include <exception>
#include "Layer.h"
#include "activationFuncs.h"

typedef MatrixXd (*ActivationFuncDer)(const MatrixXd&);


MatrixXd ReLUDer(const MatrixXd& input);

MatrixXd SigmoidDer(const MatrixXd& input);

MatrixXd TanhDer(const MatrixXd& input);

MatrixXd SoftmaxDer(const MatrixXd& input);


ActivationFuncDer find_activation_der(ActivationFunc activation_func);

class ActivationDerivativeNotFound: public std::exception {
public:
    const char* what() const noexcept override {
        return "Unknown activation function. Can't find derivative";
    }
};



#endif //DEEPDENDRO_ACTIVATIONDERIVATIVE_H
