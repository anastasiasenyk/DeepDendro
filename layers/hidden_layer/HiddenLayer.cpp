//
// Created by Yaroslav Korch on 30.03.2023.
//

#include "HiddenLayer.h"

HiddenLayer::HiddenLayer(const int curr_neurons, Shape input_shape, ActivationFunc activation) :
        biases{VectorXd::Zero(curr_neurons)},
        activ_func{activation},
        weights{MatrixXd::Random(curr_neurons, input_shape.first)},
        a_values{MatrixXd::Zero(curr_neurons, input_shape.second)}
        {
    weights /= sqrt(input_shape.first);
    shape.first = curr_neurons;
    shape.second = input_shape.second;
}


void HiddenLayer::forward_prop(const MatrixXd &prev_a_values) {
    z_values = weights * prev_a_values;
    z_values.colwise() += biases;
    a_values = activ_func(z_values);
}

inline MatrixXd HiddenLayer::calc_gradient(){
    return weights.transpose() * delta;
}

MatrixXd HiddenLayer::calc_first_back_prop(const MatrixXd &labels) {
    delta = a_values - labels;
    return calc_gradient();  // <-- gradient
}

MatrixXd HiddenLayer::calc_back_prop(const MatrixXd &gradient) {
    MatrixXd relu_derivative = find_activation_der(activ_func)(z_values);
    delta = gradient.cwiseProduct(relu_derivative);
    return calc_gradient();
}

void HiddenLayer::apply_back_prop(double learning_rate, const MatrixXd &prev_a_values) {
    auto m = static_cast<double> (delta.cols());
    weights -= learning_rate * (1. / m) * delta * prev_a_values.transpose();
    biases -= learning_rate * (1. / m) * delta.rowwise().sum();
}

const MatrixXd &HiddenLayer::getAValues() {
    return a_values;
}

const MatrixXd &HiddenLayer::getZValues() {
    return z_values;
}

void HiddenLayer::setZValues(const MatrixXd &zValues) {
    z_values = zValues;
}

void HiddenLayer::setAValues(const MatrixXd &aValues) {
    a_values = aValues;
}
