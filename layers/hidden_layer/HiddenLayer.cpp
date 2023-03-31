//
// Created by Yaroslav Korch on 30.03.2023.
//

#include "HiddenLayer.h"

HiddenLayer::HiddenLayer(){
    prev_layer = nullptr;
}

HiddenLayer::HiddenLayer(const MatrixXd &data) {
    shape.first = data.rows();
    shape.second = data.cols();
    a_values = data;
}

HiddenLayer::HiddenLayer(const int curr_neurons, HiddenLayer *ancestor, ActivationFunc activation) : prev_layer{
        ancestor},
                                                                                               biases{VectorXd::Zero(
                                                                                                       curr_neurons)},
                                                                                               activ_func{
                                                                                                       activation} {
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

void HiddenLayer::forward_prop() {
    // broadcasting the bias works ! ;)
    z_values = weights * prev_layer->a_values;
    z_values.colwise() += biases;
    a_values = activ_func(z_values);
}

void HiddenLayer::first_back_prop(double learning_rate, const VectorXd &labels) {
    MatrixXd relu_derivative = (a_values.array() > 0.0).cast<double>();
    MatrixXd delta = relu_derivative.cwiseProduct(a_values - labels);

    prev_layer->delta_next_layer = delta;


    // update weights and biases
    weights -= learning_rate * delta * prev_layer->a_values.transpose();
    biases -= learning_rate * delta.rowwise().sum();
}

void HiddenLayer::back_prop(double learning_rate) {
    MatrixXd delta = prev_layer->a_values.array() * (1 - prev_layer->a_values.array());
    delta = delta.cwiseProduct(weights.transpose() * delta_next_layer);

    prev_layer->delta_next_layer = delta;

// update weights and biases using gradients
    weights -= learning_rate * delta * prev_layer->a_values.transpose();
    biases -= learning_rate * delta.rowwise().sum();;
}


