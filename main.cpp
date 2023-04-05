//
// Created by Yaroslav Korch on 30.03.2023.
//

#include "Model.h"
#include "MNISTProcess.h"


int main() {
    srand((unsigned int) time(0));

    MNISTProcess mnistProcessTrain = MNISTProcess();
    std::map<std::string, std::pair<MatrixXd, MatrixXd>> data = mnistProcessTrain.getData("../MNIST_ORG");

    MatrixXd trainImages = data["train"].first;
    MatrixXd trainLabels = data["train"].second;

    MatrixXd testImages = data["test"].first;
    MatrixXd testLabels = data["test"].second;

    Model model;
    model.addInput(trainImages);
    model.addOutput(trainLabels);

    model.addLayer(16, activation::relu);
    model.addLayer(16, activation::relu);
    model.train(500, 0.05);

    MatrixXd predicted = model.predict(testImages);
    std::cout << model.calc_accuracy(predicted, testLabels) << std::endl;

    return 0;
}