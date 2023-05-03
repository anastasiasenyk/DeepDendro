//
// Created by Yaroslav Korch on 30.03.2023.
//

#include "Model.h"
#include "MNISTProcess.h"
#include "Filter.h"
#include "ConvLayer.h"

int main() {
    srand((unsigned int) time(0));

    ConvLayer<2, 3> ConvL{{3, 3, 3},
                          {6, 6, 3},
    };
    ConvL.convolve_all();
//    ConvL.print();


    MNISTProcess mnistProcessTrain = MNISTProcess();
    DataSets data = mnistProcessTrain.getData("../MNIST_ORG");

    Model model;
    model.addInput(data.trainData);
    model.addOutput(data.trainLabels);


    model.addLayer(128, activation::relu);
    model.addLayer(64, activation::relu);
    model.train(30, 0.05);
    model.calc_accuracy(model.predict(data.testData), data.testLabels, true);
    return 0;
}