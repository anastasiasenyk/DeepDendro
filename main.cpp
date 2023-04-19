//
// Created by Yaroslav Korch on 30.03.2023.
//

#include "Model.h"
#include "MNISTProcess.h"
#include "inter_model.h"


int main() {
//    srand((unsigned int) time(0));
    std::string pathToMNIST = "../MNIST_ORG";
    MNISTProcess mnistProcessTrain = MNISTProcess();
//
//    DataSets data = mnistProcessTrain.getData(pathToMNIST);

    tbb::concurrent_queue<std::pair<MatrixXd, MatrixXd>> mainQ;
    mnistProcessTrain.enqueueMiniBatches(64, mainQ, pathToMNIST);

//    Model model;
//    model.addInput(data.trainData);
//    model.addOutput(data.trainLabels);
//
//    model.addLayer(16, activation::relu);
//    model.addLayer(8, activation::relu);
//
//    model.train(100, 0.05);
//    model.calc_accuracy(model.predict(data.testData), data.testLabels, true);
    return 0;
}