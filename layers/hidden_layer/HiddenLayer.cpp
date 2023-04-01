//
// Created by Yaroslav Korch on 30.03.2023.
//

#include "HiddenLayer.h"

HiddenLayer::HiddenLayer(const int curr_neurons, Shape input_shape, ActivationFunc activation) :
        prev_layer{nullptr},
        biases{VectorXd::Zero(curr_neurons)},
        activ_func{activation} {
    weights = MatrixXd::Random(curr_neurons, input_shape.first);
    a_values = MatrixXd::Zero(curr_neurons, input_shape.second);
    weights /= 100;
    shape.first = curr_neurons;
    shape.second = input_shape.second;
#ifdef DEBUG
    std::cout << "Do sizes match?" << "\n";
    std::cout << "PrevLayer's shape: " << input_shape.first << " " << input_shape.second << "\n";
    std::cout << "Weights' shape: " << curr_neurons << " " << input_shape.first << "\n";
    std::cout << "Biases' shape: " << curr_neurons << "\n";
    std::cout << "Z_values' shape: " << curr_neurons << " " << input_shape.second << "\n\n";
#endif
}


HiddenLayer::HiddenLayer(const int curr_neurons, HiddenLayer *ancestor, ActivationFunc activation) :
        prev_layer{ancestor},
        biases{VectorXd::Zero(curr_neurons)},
        activ_func{activation} {
    weights = MatrixXd::Random(curr_neurons, prev_layer->shape.first);
    a_values = MatrixXd::Zero(curr_neurons, prev_layer->shape.second);
    weights /= 100;
    shape.first = curr_neurons;
    shape.second = prev_layer->shape.second;

#ifdef DEBUG
    std::cout << "Do sizes match?" << "\n";
    std::cout << "PrevLayer's shape: " << prev_layer->shape.first << " " << prev_layer->shape.second << "\n";
    std::cout << "Weights' shape: " << curr_neurons << " " << prev_layer->shape.first << "\n";
    std::cout << "Biases' shape: " << curr_neurons << "\n";
    std::cout << "Z_values' shape: " << curr_neurons << " " << prev_layer->shape.second << "\n\n";
#endif
}


void HiddenLayer::first_forward_prop(const MatrixXd &input){
    z_values = weights * input;
    z_values.colwise() += biases;
    a_values = activ_func(z_values);
}

void HiddenLayer::forward_prop() {
    first_forward_prop(prev_layer->a_values);
}

void HiddenLayer::first_back_prop(double learning_rate, const VectorXd &labels) {
    MatrixXd relu_derivative = (a_values.array() > 0.0).cast<double>();

    // TODO: It breaks here because of diff shapes --->
    MatrixXd delta = relu_derivative.cwiseProduct(a_values - labels.transpose());

    // assuming it can not be a null pointer!
    prev_layer->delta_next_layer = delta;

    // update weights and biases
    weights -= learning_rate * delta * prev_layer->a_values.transpose();
    biases -= learning_rate * delta.rowwise().sum();
}

void HiddenLayer::back_prop(double learning_rate) {
    last_back_prop(learning_rate, prev_layer->a_values);
}

void HiddenLayer::last_back_prop(double learning_rate, const MatrixXd &input){
    MatrixXd delta = input.array() * (1 - input.array());
    delta = delta.cwiseProduct(weights.transpose() * delta_next_layer);

    if (prev_layer != nullptr) prev_layer->delta_next_layer = delta;

// update weights and biases using gradients
    weights -= learning_rate * delta * input.transpose();
    biases -= learning_rate * delta.rowwise().sum();
}