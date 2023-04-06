//
// Created by Yaroslav Korch on 30.03.2023.
//

#include "HiddenLayer.h"

HiddenLayer::HiddenLayer(const int curr_neurons, Shape input_shape, ActivationFunc activation) :
        biases{VectorXd::Zero(curr_neurons)},
        activ_func{activation} {
    prev_layer = nullptr;
    weights = MatrixXd::Random(curr_neurons, input_shape.first);
    a_values = MatrixXd::Zero(curr_neurons, input_shape.second);
    weights /= sqrt(input_shape.first);
    shape.first = curr_neurons;
    shape.second = input_shape.second;
}


HiddenLayer::HiddenLayer(const int curr_neurons, const std::shared_ptr<HiddenLayer> ancestor, ActivationFunc activation) :

        biases{VectorXd::Zero(curr_neurons)},
        activ_func{activation} {
    prev_layer = ancestor;
    weights = MatrixXd::Random(curr_neurons, prev_layer->shape.first);
    a_values = MatrixXd::Zero(curr_neurons, prev_layer->shape.second);
    weights /= sqrt(ancestor->shape.first);
    shape.first = curr_neurons;
    shape.second = prev_layer->shape.second;
}


void HiddenLayer::first_forward_prop(const MatrixXd &input) {
    z_values = weights * input;
    z_values.colwise() += biases;
    a_values = activ_func(z_values);
}

void HiddenLayer::forward_prop() {
    first_forward_prop(prev_layer->a_values);
}

void HiddenLayer::first_back_prop(double learning_rate, const MatrixXd &labels) {
    MatrixXd delta = a_values - labels;

    // assuming it can not be a null pointer!
    prev_layer->weight_delta_next_layer = weights.transpose() * delta;

    // update weights and biases
    auto m = static_cast<double> (delta.cols());

    weights -= learning_rate * (1. / m) * delta * prev_layer->a_values.transpose();
    biases -= learning_rate * (1. / m) * delta.rowwise().sum();


}

void HiddenLayer::back_prop(double learning_rate) {
    last_back_prop(learning_rate, prev_layer->a_values);
}

void HiddenLayer::last_back_prop(double learning_rate, const MatrixXd &input) {
    MatrixXd relu_derivative = find_activation_der(activ_func)(z_values);
    MatrixXd delta = weight_delta_next_layer.cwiseProduct(relu_derivative);

    if (prev_layer != nullptr) {
        prev_layer->weight_delta_next_layer = weights.transpose() * delta;
    }

    // update weights and biases using gradients2
    auto m = static_cast<double> (delta.cols());
    weights -= learning_rate * (1 / m) * delta * input.transpose();
    biases -= learning_rate * (1 / m) * delta.rowwise().sum();
}

const MatrixXd &HiddenLayer::getAValues() {
    return a_values;
}
