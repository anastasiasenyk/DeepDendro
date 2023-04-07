//
// Created by Yaroslav Korch on 30.03.2023.
//

#ifndef DEEPDENDRO_HIDDENLAYER_H
#define DEEPDENDRO_HIDDENLAYER_H

#include "activationFuncs.h"
#include "activationDerivative.h"
#include "Layer.h"
#include "iostream"

class HiddenLayer : public Layer {
    MatrixXd weights;
    VectorXd biases;
    MatrixXd z_values;
    MatrixXd a_values;
    MatrixXd delta;

    ActivationFunc activ_func;

public:
    // for first layer
    HiddenLayer(int curr_neurons, Shape prev_shape, ActivationFunc activation);

    void forward_prop(const MatrixXd &prev_a_values);

    MatrixXd calc_gradient();

    MatrixXd calc_first_back_prop(const MatrixXd &labels);

    MatrixXd calc_back_prop(const MatrixXd &gradient);

    void apply_back_prop(double learning_rate, const MatrixXd &prev_a_values);

    const MatrixXd &getAValues();
};

#endif //DEEPDENDRO_HIDDENLAYER_H
