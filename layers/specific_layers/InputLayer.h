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
};


#endif //DEEPDENDRO_INPUTLAYER_H
