//
// Created by Yaroslav Korch on 30.03.2023.
//

#include "Model.h"

int main(){

    VectorXd labels= VectorXd::Zero(4);

    Model model;
    model.addInput(MatrixXd::Random(784, 4));
    model.addOutput(labels);

    model.addLayer(16, activation::relu);
    model.addLayer(8, activation::relu);
    model.train();

    return 0;
}