//
// Created by Yaroslav Korch on 30.03.2023.
//

#include "Model.h"
#include "MNISTProcess.h"
#include "inter_model.h"


int main() {
    srand((unsigned int) time(0));

    MNISTProcess mnistProcessTrain = MNISTProcess();
    DataSets data = mnistProcessTrain.getData("../MNIST_ORG");

    Model model;
    model.addInput(data.trainData);
    model.addOutput(data.trainLabels);

    model.addLayer(16, activation::relu);
    model.addLayer(16, activation::relu);
//    model.train(100, 0.05);
//    model.calc_accuracy(model.predict(data.testData), data.testLabels, true);

    InterModel models = InterModel();
    models.addModel(model, 100, 0.05);
    models.runThreads(2);

    return 0;
}