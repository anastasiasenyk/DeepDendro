//
// Created by Anastasiaa on 04.05.2023.
//

#ifndef DEEPDENDRO_CONCATENATEDLAYER_H
#define DEEPDENDRO_CONCATENATEDLAYER_H

#include "Layer.h"
#include "activationFuncs.h"
#include "activationDerivative.h"

class ConcatenatedLayer : public Layer {
    MatrixXd weights;
    VectorXd biases;
    MatrixXd z_values;
    MatrixXd a_values;
    MatrixXd delta;

    ActivationFunc activ_func;
    ActivationFuncDer activ_func_derivative;
public:
    ConcatenatedLayer(std::shared_ptr<Layer> layer1, std::shared_ptr<Layer> layer2);
    void parameters_init(){};
    void forward_prop(){};

//    MatrixXd calc_gradient();
//    MatrixXd calc_back_prop(const MatrixXd &gradient);
//    void apply_back_prop(double learning_rate);
    MatrixXd getAValues() const;

    double calc_accuracy(){ return 0; };

};


#endif //DEEPDENDRO_CONCATENATEDLAYER_H
