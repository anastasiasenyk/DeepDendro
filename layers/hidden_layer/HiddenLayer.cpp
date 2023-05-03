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
    weights = MatrixXd::Random(num_neurons, parents[0]->get_shape().back()); // TODO: change second param for general (conv layer)
    weights /= sqrt(parents[0]->get_shape().back());
//    a_values = MatrixXd::Zero(num_neurons, input_size); // TODO: add init
}

//void HiddenLayer::first_forward_prop(const MatrixXd &input) {
//    z_values = weights * input;
//    z_values.colwise() += biases;
//    a_values = activ_func(z_values);
//}
//
//void HiddenLayer::forward_prop() {
//    first_forward_prop(prev_layer->a_values);
//}

//void HiddenLayer::first_back_prop(double learning_rate, const MatrixXd &labels) {
//    MatrixXd delta = a_values - labels;
//
//    // assuming it can not be a null pointer!
//    prev_layer->weight_delta_next_layer = weights.transpose() * delta;
//
//    // update weights and biases
//    auto m = static_cast<double> (delta.cols());
//
//    weights -= learning_rate * (1. / m) * delta * prev_layer->a_values.transpose();
//    biases -= learning_rate * (1. / m) * delta.rowwise().sum();
//}
//
//void HiddenLayer::back_prop(double learning_rate) {
//    last_back_prop(learning_rate, prev_layer->a_values);
//}
//
//void HiddenLayer::last_back_prop(double learning_rate, const MatrixXd &input) {
//    MatrixXd relu_derivative = find_activation_der(activ_func)(z_values);
//    MatrixXd delta = weight_delta_next_layer.cwiseProduct(relu_derivative);
//
//    if (prev_layer != nullptr) {
//        prev_layer->weight_delta_next_layer = weights.transpose() * delta;
//    }
//
//    // update weights and biases using gradients2
//    auto m = static_cast<double> (delta.cols());
//    weights -= learning_rate * (1 / m) * delta * input.transpose();
//    biases -= learning_rate * (1 / m) * delta.rowwise().sum();
//}

//const MatrixXd &HiddenLayer::getAValues() {
//    return a_values;
//}
