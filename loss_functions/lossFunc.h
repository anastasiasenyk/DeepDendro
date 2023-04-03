//
// Created by Yaroslav Korch on 30.03.2023.
//

#ifndef DEEPDENDRO_LOSSFUNC_H
#define DEEPDENDRO_LOSSFUNC_H
#include "Layer.h"
class lossFunc {
public:
    double crossEntropy(const MatrixXd& predict, const MatrixXd& Y);
};


#endif //DEEPDENDRO_LOSSFUNC_H
