//
// Created by Yaroslav Korch on 30.03.2023.
//

#ifndef DEEPDENDRO_HIDDENLAYER_H
#define DEEPDENDRO_HIDDENLAYER_H

#include "actiovation_funcs.h"
#include "Layer.h"
#include "iostream"

class HiddenLayer : public Layer {
    MatrixXd weights;
    VectorXd biases;
    MatrixXd z_values;
    MatrixXd a_values;
    ActivationFunc activ_func;

    MatrixXd delta_next_layer;
    HiddenLayer * prev_layer;

public:
    HiddenLayer();
    HiddenLayer(const MatrixXd &data);

    HiddenLayer(int curr_neurons, HiddenLayer *ancestor, ActivationFunc activation);

    void forward_prop();

    void first_back_prop(double learning_rate, const VectorXd &labels);

    void back_prop(double learning_rate);
};


#endif //DEEPDENDRO_HIDDENLAYER_H
