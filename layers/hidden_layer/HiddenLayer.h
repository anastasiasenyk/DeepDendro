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

//    void first_forward_prop(const MatrixXd &input);
//
//    void forward_prop();
//
//    void first_back_prop(double learning_rate, const MatrixXd &labels);
//
//    void back_prop(double learning_rate);
//
//    void last_back_prop(double learning_rate, const MatrixXd &a_values);
//
//    const MatrixXd &getAValues();
};

#endif //DEEPDENDRO_HIDDENLAYER_H
