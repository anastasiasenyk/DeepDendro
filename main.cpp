//
// Created by Yaroslav Korch on 30.03.2023.
//

#include "Model.h"
#include "MNISTProcess.h"
#include "Filter.h"
#include "Pooling.h"
#include "Convolutions.h"
#include "FlatteningLayer.h"
#include "CIFAR10_Reader.h"





int main() {
    srand((unsigned int) time(0));

    std::string dir_cifar_path = "../CIFAR10/cifar-10-batches-bin";
    auto [train_data, test_data]  = load_cifar10_whole(dir_cifar_path);
    auto [train_images, train_labels] = train_data;
    auto [test_images, test_labels] = test_data;


    MNISTProcess mnistProcessTrain = MNISTProcess();
    DataSets data = mnistProcessTrain.getData("../MNIST_ORG");
    Model model;
    model.addInput(data.trainData);
    model.addOutput(data.trainLabels);

    model.addDense(32, activation::relu);

    bool verbose = true;
    model.train(200, 0.005, verbose);
    model.calc_accuracy(model.predict(data.testData), data.testLabels, verbose);

    return 0;
}