//
// Created by Yaroslav Korch on 30.03.2023.
//


#include "HiddenLayer.h"

HiddenLayer::HiddenLayer(long curr_neurons, activation type) :
        Layer(curr_neurons),
        biases(), weights(), z_values(), a_values() {
    activ_func = find_activation_func(type);
    activ_func_derivative = find_activation_der(activ_func);
}

void HiddenLayer::parameters_init() {
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

void HiddenLayer::forward_prop() {
    z_values = weights * get_parents().front()->getAValues();
    z_values.colwise() += biases;
    a_values = activ_func(z_values);
}

MatrixXd HiddenLayer::getAValues() const {
    return a_values;
}

inline MatrixXd HiddenLayer::calc_gradient() {
    return weights.transpose() * delta;
}

MatrixXd HiddenLayer::calc_back_prop(const MatrixXd &gradient) {
    MatrixXd relu_derivative = activ_func_derivative(z_values);
    delta = gradient.cwiseProduct(relu_derivative);
    return calc_gradient();
}

void HiddenLayer::apply_back_prop(double learning_rate) {
    auto m = static_cast<double> (delta.cols());
    weights -= learning_rate * (1. / m) * delta * get_parents().front()->getAValues().transpose();
    biases -= learning_rate * (1. / m) * delta.rowwise().sum();
}
