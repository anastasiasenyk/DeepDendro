//
// Created by Anastasiaa on 05.04.2023.
//

#ifndef DEEPDENDRO_DATASETS_H
#define DEEPDENDRO_DATASETS_H

#endif //DEEPDENDRO_DATASETS_H

#include "Layer.h"

struct DataSets {
    MatrixXd testData;
    MatrixXd testLabels;
    MatrixXd trainData;
    MatrixXd trainLabels;
} ;

struct TrainingSet {
    MatrixXd trainData;
    MatrixXd trainLabels;
};

struct TestSet {
    MatrixXd testData;
    MatrixXd testLabels;
};