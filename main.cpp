//
// Created by Yaroslav Korch on 30.03.2023.
//

#include "Model.h"
#include "MNISTProcess.h"
#include "Filter.h"
#include "Pooling.h"
#include "Convolutions.h"

int main() {
    srand((unsigned int) time(0));

    MNISTProcess mnistProcessTrain = MNISTProcess();
    DataSets data = mnistProcessTrain.getData("../MNIST_ORG");

    Eigen::TensorMap<Eigen::Tensor<double, 3>> input3d(data.trainData.data(), data.trainData.rows(),
                                                       data.trainData.cols(), 1);


    Model model;
    model.addInput(data.trainData);
    model.addOutput(data.trainLabels);


    model.calc_accuracy(model.predict(data.testData), data.testLabels, true);
    return 0;
}