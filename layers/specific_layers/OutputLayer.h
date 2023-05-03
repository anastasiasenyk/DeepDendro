//
// Created by Anastasiaa on 03.05.2023.
//

#ifndef DEEPDENDRO_OUTPUTLAYER_H
#define DEEPDENDRO_OUTPUTLAYER_H

#include "activationFuncs.h"
#include "activationDerivative.h"
#include "activationFuncs.h"
#include "Layer.h"

class OutputLayer : public Layer {
    MatrixXd weights;
    VectorXd biases;
    MatrixXd z_values;
    MatrixXd a_values;
    MatrixXd delta;

    ActivationFunc activ_func;
    ActivationFuncDer activ_func_derivative;

    MatrixXd weight_delta_next_layer_;

    MatrixXd train_labels;
    MatrixXd predict_after_forward_prop();
public:
    OutputLayer(const MatrixXd& train_labels, activation type);
    void parameters_init();
    void forward_prop();
    void back_prop(double learning_rate);

    MatrixXd calc_gradient();

    MatrixXd calc_first_back_prop();

    void apply_back_prop(double learning_rate);

    MatrixXd getAValues() const;

    double calc_accuracy();

};



#endif //DEEPDENDRO_OUTPUTLAYER_H
