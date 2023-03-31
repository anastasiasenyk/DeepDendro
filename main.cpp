//
// Created by Yaroslav Korch on 30.03.2023.
//

#include "Model.h"

int main(){

    VectorXd labels = VectorXd::Zero(4);
    labels[0] = 1; labels[1] = 1; labels[2] = 1; labels[3] = 0;


    Model model;
    model.addInput(MatrixXd::Random(784, 4));
    model.addOutput(labels);

    model.addLayer(16, activation::relu);
    model.addLayer(8, activation::relu);
    model.train();

    return 0;
}