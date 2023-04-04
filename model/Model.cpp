//
// Created by Yaroslav Korch on 30.03.2023.
//

#include "Model.h"


Model::Model() {
    layers.reserve(10);
    save_prev_layer = nullptr;
//    train_data = MatrixXd::Zero(1, 1);
//    train_labels = VectorXd::Zero(1);
}

void Model::addInput(const MatrixXd &data) {
    train_data = data;
}

void Model::addOutput(const MatrixXd &labels) {
    train_labels = labels;
}

void Model::addLayer(int neurons, activation activationType) {
    if (save_prev_layer != nullptr) {
        layers.emplace_back(neurons, save_prev_layer, find_activation_func(activationType));
        save_prev_layer = &layers.back();
        return;
    }
    Shape prev_shape = {train_data.rows(), train_data.cols()};
    layers.emplace_back(neurons, prev_shape, find_activation_func(activationType));
    save_prev_layer = &layers.back();
}


void Model::train(size_t epochs, double learning_rate) {
    int j;
    addLayer(train_labels.rows(), activation::sigmoid);
    for (size_t i = 0; i < epochs; ++i) {

        // first forward prop
        layers[0].first_forward_prop(train_data);
        for (j = 1; j < layers.size();) {
            layers[j++].forward_prop();
        }
#ifdef DEBUG
        double lossRes = lossFunc().crossEntropy(layers[layers.size() - 1].getAValues(), train_labels);
        std::cout << "After epoch: " << i << " cost: " << lossRes << std::endl;
#endif
        // first back_prop
        layers.back().first_back_prop(learning_rate, train_labels);
        // all other back props
        for (j = layers.size() - 2; j > 0;) {
            layers[j--].back_prop(learning_rate);
        }
        layers[0].last_back_prop(learning_rate, train_data);
    }
}

