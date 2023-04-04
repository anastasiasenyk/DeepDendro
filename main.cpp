//
// Created by Yaroslav Korch on 30.03.2023.
//

#include "Model.h"

int main() {
    srand((unsigned int) time(0));

    MatrixXd labels = MatrixXd::Random(2, 4);
    labels = (labels.array() > 0).cast<double>();
    std::cout << labels << std::endl;

    Model model;
    model.addInput(MatrixXd::Random(784, 4));
    model.addOutput(labels);

    model.addLayer(16, activation::relu);
    model.addLayer(8, activation::relu);
    model.train(10000, 0.005);

    return 0;
}