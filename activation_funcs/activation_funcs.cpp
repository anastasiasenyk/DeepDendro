//
// Created by Yaroslav Korch on 30.03.2023.
//

#include "actiovation_funcs.h"


MatrixXd ReLU(const MatrixXd& input){
    return input.cwiseMax(0);
}

ActivationFunc find_activation_func(activation type){
    switch (type){
        case relu:
            return ReLU;
        default:
            throw ActivationNotFound();
    }
}
