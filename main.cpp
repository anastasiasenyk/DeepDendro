//
// Created by Yaroslav Korch on 30.03.2023.
//

#include "Model.h"

int main() {
    srand((unsigned int) time(0));

    MatrixXd labels = MatrixXd::Random(1, 4);
    labels = (labels.array() > 0).cast<double>();
    std::cout << labels << std::endl;

    Model model;
    model.addInput(MatrixXd::Random(784, 4));
    model.addOutput(labels);

    model.addLayer(16, activation::relu);
    model.addLayer(8, activation::relu);
    model.train(1000, 0.05);

    return 0;
}