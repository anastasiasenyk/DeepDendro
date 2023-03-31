//
// Created by Yaroslav Korch on 30.03.2023.
//

#include "Model.h"



void Model::addInput(const MatrixXd &data) {
    train_data = HiddenLayer{data};
    save_prev_layer = &train_data;
}

void Model::addLayer(int neurons, activation activationType) {
    layers.push_back(HiddenLayer{neurons, save_prev_layer, find_activation_func(activationType)});
    save_prev_layer = &layers.back();
}

void Model::addOutput(const VectorXd &labels) {
    train_labels = labels;
}

void Model::train(size_t epochs, double learning_rate) {
    for (size_t i = 0; i < epochs; ++i) {
        for (auto &layer: layers) {
            layer.forward_prop();
        }
        // first back_prop
        layers.back().first_back_prop(learning_rate, train_labels);
        // all other back props
        for (int j = layers.size() - 2; j > -1;) {
            layers[j--].back_prop(learning_rate);
        }
    }
}

