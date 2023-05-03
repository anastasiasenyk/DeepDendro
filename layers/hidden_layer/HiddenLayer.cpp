//
// Created by Yaroslav Korch on 30.03.2023.
//

#include "HiddenLayer.h"

HiddenLayer::HiddenLayer(const int curr_neurons, MShape input_shape, activation type) :
        biases{VectorXd::Zero(curr_neurons)},
        weights{MatrixXd::Random(curr_neurons, input_shape.first)},
        a_values{MatrixXd::Zero(curr_neurons, input_shape.second)} {

    auto [activFunc, activDer] = find_activation_func_DENSE(type);
    activ_func = activFunc; activ_func_derivative = activDer;

    weights *= sqrt(2 / static_cast<double>(input_shape.first));
    shape.first = curr_neurons;
    shape.second = input_shape.second;
}


void HiddenLayer::forward_prop(const MatrixXd &prev_a_values) {
    z_values = weights * prev_a_values;
    z_values.colwise() += biases;
    a_values = activ_func(z_values);
}

inline MatrixXd HiddenLayer::calc_gradient() {
    return weights.transpose() * delta;
}

MatrixXd HiddenLayer::calc_first_back_prop(const MatrixXd &labels) {
    delta = a_values - labels;
    return calc_gradient();
}

MatrixXd HiddenLayer::calc_back_prop(const MatrixXd &gradient) {
    MatrixXd relu_derivative = activ_func_derivative(z_values);
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
