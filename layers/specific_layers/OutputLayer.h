//
// Created by Anastasiaa on 03.05.2023.
//

#ifndef DEEPDENDRO_OUTPUTLAYER_H
#define DEEPDENDRO_OUTPUTLAYER_H

#include "activationFuncs.h"
#include "Layer.h"

class OutputLayer : public Layer {
    MatrixXd weights;
    VectorXd biases;
    MatrixXd z_values;
    MatrixXd a_values;
    ActivationFunc activ_func;

    MatrixXd weight_delta_next_layer_;

    MatrixXd train_labels;
public:
    OutputLayer(MatrixXd train_labels, ActivationFunc activation);
    void parameters_init();
};



#endif //DEEPDENDRO_OUTPUTLAYER_H
