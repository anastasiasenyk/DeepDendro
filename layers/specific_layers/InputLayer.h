//
// Created by Anastasiaa on 03.05.2023.
//

#ifndef DEEPDENDRO_INPUTLAYER_H
#define DEEPDENDRO_INPUTLAYER_H

#include "activationFuncs.h"
#include "Layer.h"


class InputLayer : public Layer {
    MatrixXd a_values;
public:
    InputLayer(MatrixXd &data);
    void parameters_init();
    void forward_prop(){};
    void back_prop(double learning_rate){};
    MatrixXd getAValues() const;
    double calc_accuracy(){ return 0; };
    MatrixXd calc_gradient(){
        return MatrixXd::Zero(0, 0);};;
    MatrixXd calc_back_prop(const MatrixXd &gradient){
        return MatrixXd::Zero(0, 0);};
    void apply_back_prop(double learning_rate){};
};


#endif //DEEPDENDRO_INPUTLAYER_H
