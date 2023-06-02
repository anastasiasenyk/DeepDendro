//
// Created by Anastasiaa on 05.04.2023.
//

#ifndef DEEPDENDRO_DATASETS_H
#define DEEPDENDRO_DATASETS_H

#endif //DEEPDENDRO_DATASETS_H

#include "Layer.h"

typedef Eigen::Tensor<double, 3> Tensor3d;
typedef Eigen::Tensor<double, 2> Tensor2d;

struct DataSets {
    Tensor3d testData;
    MatrixXd testLabels;
    Tensor3d trainData;
    MatrixXd trainLabels;
} ;