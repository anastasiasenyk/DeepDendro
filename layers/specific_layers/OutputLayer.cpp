//
// Created by Anastasiaa on 03.05.2023.
//

#include "OutputLayer.h"

OutputLayer::OutputLayer(MatrixXd train_labels, ActivationFunc activation) :
    activ_func(activation),
    Layer({train_labels.rows(), train_labels.cols()}){
}

void OutputLayer::parameters_init() {
    long num_neurons = get_shape().front();
    std::vector<std::shared_ptr<Layer>> parents = get_parents();

    if (parents.size() > 1) {
        throw std::logic_error("Layer has more than one parent. Only half-orphans are allowed.");
    }

    if (parents.empty()) {
        throw std::logic_error("Layer should have one parent.");
    }

    biases = VectorXd::Zero(static_cast<long>(num_neurons));
    weights = MatrixXd::Random(num_neurons, parents[0]->get_shape().back()); // TODO: change second param for general (conv layer)
    weights /= sqrt(parents[0]->get_shape().back());
//    a_values = MatrixXd::Zero(num_neurons, input_size); // TODO: add init
}