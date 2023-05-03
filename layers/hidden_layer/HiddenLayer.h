//
// Created by Yaroslav Korch on 30.03.2023.
//

#ifndef DEEPDENDRO_HIDDENLAYER_H
#define DEEPDENDRO_HIDDENLAYER_H

#include "iostream"
#include "Layer.h"
#include "activationFuncs.h"
#include "activationDerivative.h"


typedef std::pair<int, int> Shape;

class HiddenLayer : public Layer {
    MatrixXd weights;
    VectorXd biases;
    MatrixXd z_values;
    MatrixXd a_values;
    MatrixXd delta;

    ActivationFunc activ_func;
    ActivationFuncDer activ_func_derivative;

    MatrixXd weight_delta_next_layer_;
public:
    HiddenLayer(long curr_neurons, activation activation);

    void parameters_init();

    void forward_prop();

//    void back_prop(double learning_rate);

    MatrixXd calc_gradient();

    MatrixXd calc_back_prop(const MatrixXd &gradient);

    void apply_back_prop(double learning_rate);

    MatrixXd getAValues() const;
    double calc_accuracy(){ return 0; };
};

#endif //DEEPDENDRO_HIDDENLAYER_H
