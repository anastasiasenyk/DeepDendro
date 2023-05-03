//
// Created by Anastasiaa on 03.05.2023.
//

#include "OutputLayer.h"

OutputLayer::OutputLayer(const MatrixXd& train_labels, ActivationFunc activation) :
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
    weights = MatrixXd::Random(num_neurons, parents[0]->get_shape().front());
    weights /= sqrt(parents[0]->get_shape().front());
    a_values = MatrixXd::Zero(num_neurons, parents[0]->getAValues().cols());
}

void OutputLayer::forward_prop() {
    z_values = weights * get_parents().front()->getAValues();
    z_values.colwise() += biases;
    a_values = activ_func(z_values);
}

void OutputLayer::back_prop(double learning_rate) {
    MatrixXd delta = a_values - train_labels;

    get_parents().front()->weight_delta_next_layer_ = weights.transpose() * delta;

    auto m = static_cast<double> (delta.cols());

    weights -= learning_rate * (1. / m) * delta * get_parents().front()->a_values.transpose();
    biases -= learning_rate * (1. / m) * delta.rowwise().sum();
}

MatrixXd OutputLayer::getAValues() const {
    return a_values;
}