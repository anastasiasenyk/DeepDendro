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


    ActivationFunc activ_func;

    MatrixXd weight_delta_next_layer;
    // TODO: rewrite so this pointer will be shared
    HiddenLayer *prev_layer;

public:
    // for first layer
    HiddenLayer(int curr_neurons, Shape prev_shape, ActivationFunc activation);

    // for further layers
    HiddenLayer(int curr_neurons, HiddenLayer *ancestor, ActivationFunc activation);


    void first_forward_prop(const MatrixXd &input);

    void forward_prop();

    void first_back_prop(double learning_rate, const MatrixXd &labels);

    void back_prop(double learning_rate);

    void last_back_prop(double learning_rate, const MatrixXd &a_values);

    const MatrixXd &getAValues();
};

#endif //DEEPDENDRO_HIDDENLAYER_H
