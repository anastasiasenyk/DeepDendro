//
// Created by Yaroslav Korch on 30.03.2023.
//

#include "Model.h"


Model::Model() {
    layers.reserve(10);
    save_prev_layer = nullptr;
}

void Model::addInput(const MatrixXd &data) {
    train_data = data;
}

void Model::addOutput(const MatrixXd &labels) {
    train_labels = labels;
}

void Model::addLayer(int neurons, activation activationType) {
    if (save_prev_layer != nullptr) {
        layers.emplace_back(neurons, save_prev_layer, find_activation_func(activationType));
        save_prev_layer = &layers.back();
        return;
    }
    Shape prev_shape = {train_data.rows(), train_data.cols()};
    layers.emplace_back(neurons, prev_shape, find_activation_func(activationType));
    save_prev_layer = &layers.back();
}


void Model::train(size_t epochs, double learning_rate) {

#ifdef LOGGING
    show_console_cursor(false);
    const std::string out_of_all = " / " + std::to_string(epochs) + " ";
    ProgressBar bar{
            indicators::option::BarWidth{50},
            indicators::option::Start{"["},
            indicators::option::Fill{"■"},
            indicators::option::Lead{"■"},
            indicators::option::Remainder{"-"},
            indicators::option::End{" ]"},

            indicators::option::PrefixText{"DeepDendro Epoch: _ "},
            indicators::option::PostfixText{"Loss function: _"},
            indicators::option::ShowElapsedTime{true},
            indicators::option::ShowRemainingTime{true},
            indicators::option::ForegroundColor{Color::red},
            indicators::option::FontStyles{std::vector<FontStyle>{FontStyle::bold}},
            indicators::option::MaxProgress{epochs}
    };
    int when_calc_accuracy = 25;
    double accuracy = 0;
#endif
//    int j;
    addLayer(train_labels.rows(), activation::softmax);
    for (size_t i = 0; i < epochs; ++i) {

        // first forward prop
        layers[0].first_forward_prop(train_data);
        for (int k = 1; k < layers.size();) {
            layers[k++].forward_prop();
        }

#ifdef LOGGING
        if (i % when_calc_accuracy == 0) {
            accuracy = calc_accuracy(predict_after_forward_prop(), train_labels) * 100;
        }

        bar.set_option(indicators::option::PrefixText{"DeepDendro epoch: " + std::to_string(i + 1) + out_of_all});
        bar.tick();
        bar.set_option(indicators::option::PostfixText{
                "Loss function: " + std::to_string(lossFunc().categoryCrossEntropy(layers.back().getAValues(), train_labels)) + ", Accuracy: " + std::to_string(accuracy) + "%"});
#endif

        // first back_prop
        layers.back().first_back_prop(learning_rate, train_labels);
        // all other back props
        for (int j = layers.size() - 2; j > 0;) {
            layers[j--].back_prop(learning_rate);
        }
        layers[0].last_back_prop(learning_rate, train_data);
    }
}

MatrixXd Model::predict_after_forward_prop() {
    MatrixXd predicted_values = layers.back().getAValues();

    Eigen::Index numCols = predicted_values.cols();

    for (Eigen::Index i=0; i<numCols; i++) {
        int maxRowIndex;
        predicted_values.col(i).maxCoeff(&maxRowIndex);
        predicted_values.col(i).setZero();
        predicted_values(maxRowIndex, i) = 1;
    }
    return predicted_values;
}

double Model::calc_accuracy(const MatrixXd &predicted, const MatrixXd &true_labels) {
    double num_samples = predicted.cols();

    MatrixXd diff = (predicted - true_labels).cwiseAbs2();


    VectorXd col_sums = diff.colwise().sum();
    double num_identical_cols = (col_sums.array() == 0).count();

    return num_identical_cols / num_samples;
}