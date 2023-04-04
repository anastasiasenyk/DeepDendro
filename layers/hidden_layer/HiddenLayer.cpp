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

    // update weights and biases
    auto m = static_cast<double> (delta.cols());
    weights -= learning_rate * (1 / m) * delta * prev_layer->a_values.transpose();
    biases -= learning_rate * (1 / m) * delta.rowwise().sum();

    // assuming it can not be a null pointer!
    prev_layer->weight_delta_next_layer = weights.transpose() * delta;
}

void HiddenLayer::back_prop(double learning_rate) {
    last_back_prop(learning_rate, prev_layer->a_values);
}

void HiddenLayer::last_back_prop(double learning_rate, const MatrixXd &input) {
//    MatrixXd relu_derivative = (z_values.array() > 0.0).cast<double>();
    MatrixXd relu_derivative = find_activation_der(activ_func)(z_values);
    MatrixXd delta = weight_delta_next_layer.cwiseProduct(relu_derivative);

    // update weights and biases using gradients2
    auto m = static_cast<double> (delta.cols());
    weights -= learning_rate * (1 / m) * delta * input.transpose();
    biases -= learning_rate * (1 / m) * delta.rowwise().sum();

    if (prev_layer != nullptr) {
        prev_layer->weight_delta_next_layer = weights.transpose() * delta;
    }
}

const MatrixXd &HiddenLayer::getAValues() {
    return a_values;
}
