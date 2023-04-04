//
// Created by Yaroslav Korch on 30.03.2023.
//

#ifndef DEEPDENDRO_DATAPROCESSING_H
#define DEEPDENDRO_DATAPROCESSING_H


#include "Layers.h"

class DataProcessing {
public:
    static MatrixXd flatten(const MatrixXd &data);
};


#endif //DEEPDENDRO_DATAPROCESSING_H
