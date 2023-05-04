//
// Created by Anastasiaa on 03.05.2023.
//


#include "OutputLayer.h"

OutputLayer::OutputLayer(const MatrixXd &train_labels, activation type) :
        train_labels(train_labels),
        Layer({train_labels.rows(), train_labels.cols()}) {
    activ_func = find_activation_func(type);
    activ_func_derivative = find_activation_der(activ_func);
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

MatrixXd OutputLayer::getAValues() const {
    return a_values;
}

MatrixXd OutputLayer::getTrainLabels() const {
    return train_labels;
}

MatrixXd OutputLayer::predict_after_forward_prop() {
    MatrixXd predicted_values = a_values;
    Eigen::Index numCols = predicted_values.cols();
    int maxRowIndex;
    for (Eigen::Index i = 0; i < numCols; i++) {
        predicted_values.col(i).maxCoeff(&maxRowIndex);
        predicted_values.col(i).setZero();
        predicted_values(maxRowIndex, i) = 1;
    }
    return predicted_values;
}

double OutputLayer::calc_accuracy() {
    MatrixXd predicted = predict_after_forward_prop();
    double num_samples = predicted.cols();

    MatrixXd diff = (predicted - train_labels).cwiseAbs2();

    VectorXd col_sums = diff.colwise().sum();
    double num_identical_cols = (col_sums.array() == 0).count();

    return num_identical_cols / num_samples;
}

inline MatrixXd OutputLayer::calc_gradient() {
    return weights.transpose() * delta;
}

MatrixXd OutputLayer::calc_first_back_prop() {
    delta = a_values - train_labels;
    return calc_gradient();
}

void OutputLayer::apply_back_prop(double learning_rate) {
    auto m = static_cast<double> (delta.cols());
    weights -= learning_rate * (1. / m) * delta * get_parents().front()->getAValues().transpose();
    biases -= learning_rate * (1. / m) * delta.rowwise().sum();
}

