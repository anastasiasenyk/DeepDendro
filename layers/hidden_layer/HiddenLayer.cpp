//
// Created by Yaroslav Korch on 30.03.2023.
//

#include "HiddenLayer.h"

HiddenLayer::HiddenLayer(long curr_neurons, ActivationFunc activation) :
        Layer(curr_neurons),
        activ_func(activation),
        biases(), weights(), z_values(), a_values() {
}

void HiddenLayer::parameters_init() {
    long num_neurons = get_shape().back();
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

void HiddenLayer::forward_prop() {
    z_values = weights * get_parents().front()->getAValues();
    z_values.colwise() += biases;
    a_values = activ_func(z_values);
}

void HiddenLayer::back_prop(double learning_rate) {
    MatrixXd delta = a_values - get_parents().front()->a_values;

    get_parents().front()->weight_delta_next_layer_ = weights.transpose() * delta;

    auto m = static_cast<double> (delta.cols());

    weights -= learning_rate * (1. / m) * delta * get_parents().front()->a_values.transpose();
    biases -= learning_rate * (1. / m) * delta.rowwise().sum();
}

MatrixXd HiddenLayer::getAValues() const {
    return a_values;
}
