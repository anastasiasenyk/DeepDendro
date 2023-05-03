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

    MatrixXd data = MatrixXd::Random(200, 1000);
    MatrixXd label1 = MatrixXd::Random(2, 1);
//    MatrixXd label2 = MatrixXd::Random(3, 1);

    shared_layer first_layer = std::make_shared<InputLayer>(data);
//    shared_layer second_layer = std::make_shared<HiddenLayer>(20, find_activation_func(activation::softmax));
    shared_layer third_layer = std::make_shared<HiddenLayer>(10, find_activation_func(activation::relu));
    shared_layer output_1 = std::make_shared<OutputLayer>(label1, find_activation_func(activation::relu));
//    shared_layer output_2 = std::make_shared<OutputLayer>(label2, find_activation_func(activation::relu));

//    (*second_layer)(first_layer);
    (*third_layer)(first_layer);
    (*output_1)(third_layer);
//    (*output_2)(second_layer);

    Model model = Model();
    model.save(first_layer, {output_1});

    model.compile();
    model.train(1000, 0.05);

    return 0;
}