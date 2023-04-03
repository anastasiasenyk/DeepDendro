//
// Created by Yaroslav Korch on 30.03.2023.
//

#include "lossFunc.h"
#include <iostream>

double lossFunc::crossEntropy(const MatrixXd& predict, const MatrixXd& Y) {

    int m = Y.cols();
    MatrixXd cost_mat = (-1 * Y.array() * log(predict.array())).matrix() -
            ((1 - Y.array()).matrix().cwiseProduct(log1p(-predict.array()).matrix()));
    double cost = cost_mat.sum() / m;

    return cost;
}