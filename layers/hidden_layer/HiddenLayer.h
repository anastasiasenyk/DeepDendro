//
// Created by Yaroslav Korch on 30.03.2023.
//

#ifndef DEEPDENDRO_HIDDENLAYER_H
#define DEEPDENDRO_HIDDENLAYER_H

#include "iostream"
#include "Layer.h"
#include "activationFuncs.h"


typedef std::pair<int, int> Shape;

class HiddenLayer : public Layer {
    MatrixXd weights;
    VectorXd biases;
    MatrixXd z_values;
    MatrixXd a_values;
    ActivationFunc activ_func;

    MatrixXd weight_delta_next_layer_;
public:
    HiddenLayer(long curr_neurons, ActivationFunc activation);

    void parameters_init();

    void forward_prop();

    void back_prop(double learning_rate);

    MatrixXd getAValues() const;
};

#endif //DEEPDENDRO_HIDDENLAYER_H
