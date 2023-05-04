//
// Created by Yaroslav Korch on 30.03.2023.
//

#include <iostream>

//#include "activationFuncs.h"
#include "OutputLayer.h"
#include "InputLayer.h"
#include "HiddenLayer.h"
#include "Layer.h"
#include "Model.h"
#include "MNISTProcess.h"


int main() {
    typedef std::shared_ptr<Layer> shared_layer;

    MNISTProcess mnistProcessTrain = MNISTProcess();
    DataSets data = mnistProcessTrain.getData("../MNIST_ORG");

    shared_layer input = std::make_shared<InputLayer>(data.testData);
    shared_layer second_layer = std::make_shared<HiddenLayer>(16, activation::relu);
    shared_layer third_layer = std::make_shared<HiddenLayer>(8, activation::relu);
    shared_layer fourth_layer = std::make_shared<HiddenLayer>(8, activation::relu);
    shared_layer output_1 = std::make_shared<OutputLayer>(data.testLabels, activation::softmax);
    shared_layer output_2 = std::make_shared<OutputLayer>(data.testLabels, activation::softmax);


    (*second_layer)(input);
    (*third_layer)(input);
    (*fourth_layer)(third_layer);
    (*output_1)(second_layer);
    (*output_2)(fourth_layer);

    Model model = Model();
    model.save(input, {output_1, output_2});
    model.train(1000, 0.05);

    return 0;
}