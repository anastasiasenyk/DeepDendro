//
// Created by Yaroslav Korch on 30.03.2023.
//

#include "Model.h"


Model::Model() {
    dense_layers.reserve(10);
}

void Model::addInput(const MatrixXd &data) {
    train_data = data;
}

void Model::addOutput(const MatrixXd &labels) {
    train_labels = labels;
}

void Model::addDense(int neurons, activation activationType) {
    MShape prev_shape;
    if (dense_layers.empty()) {
        prev_shape = {train_data.rows(), train_data.cols()};
    } else {
        prev_shape = dense_layers.back().shape;
    }

    dense_layers.emplace_back(neurons, prev_shape, activationType);
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
                indicators::option::ForegroundColor{Color::blue},
                indicators::option::FontStyles{std::vector<FontStyle>{FontStyle::bold}},
                indicators::option::MaxProgress{epochs}

    };
    double accuracy = 0;
#endif


    MatrixXd gradient;
    addDense(train_labels.rows(), activation::softmax);
    for (size_t i = 0; i < epochs; ++i) {
        // first forward prop
        dense_layers[0].forward_prop(train_data);
        for (int k = 1; k < dense_layers.size(); k++) {
            dense_layers[k].forward_prop(dense_layers[k - 1].getAValues());
        }

#ifdef LOGGING
    accuracy = calc_accuracy(predict_after_forward_prop(), train_labels) * 100;
    bar.set_option(indicators::option::PrefixText{"DeepDendro epoch: " + std::to_string(i + 1) + out_of_all});
    bar.tick();
    bar.set_option(indicators::option::PostfixText{
            "Loss function: " +
            std::to_string(lossFunc().categoryCrossEntropy(dense_layers.back().getAValues(), train_labels)) +
            ", Accuracy: " + std::to_string(accuracy) + "%"});

#endif

        // calc back_prop
        gradient = dense_layers.back().calc_first_back_prop(train_labels);
        // all other back props
        for (int j = dense_layers.size() - 2; j > -1; j--) {
            gradient = dense_layers[j].calc_back_prop(gradient);
        }

        // here order doesn't really matter
        dense_layers.front().apply_back_prop(learning_rate, train_data);
        for (int j = 1; j < dense_layers.size(); j++) {
            dense_layers[j].apply_back_prop(learning_rate, dense_layers[j - 1].getAValues());
        }
    }
}

MatrixXd Model::predict_after_forward_prop() {
    // TODO: Use vectorization
    MatrixXd predicted_values = dense_layers.back().getAValues();
    Eigen::Index numCols = predicted_values.cols();
    int maxRowIndex;
    for (Eigen::Index i = 0; i < numCols; i++) {
        predicted_values.col(i).maxCoeff(&maxRowIndex);
        predicted_values.col(i).setZero();
        predicted_values(maxRowIndex, i) = 1;
    }
    return predicted_values;
}

MatrixXd Model::predict(const MatrixXd &testData) {
    dense_layers[0].forward_prop(testData);
    for (int k = 1; k < dense_layers.size(); k++) {
        dense_layers[k].forward_prop(dense_layers[k - 1].getAValues());
    }
    return predict_after_forward_prop();
}

double Model::calc_accuracy(const MatrixXd &predicted, const MatrixXd &true_labels, bool verbose) {
    double num_samples = predicted.cols();

    MatrixXd diff = (predicted - true_labels).cwiseAbs2();

    VectorXd col_sums = diff.colwise().sum();
    double num_identical_cols = (col_sums.array() == 0).count();

    if (verbose) {
        std::cout << YELLOW << "Test accuracy: " << 100 * num_identical_cols / num_samples << "%" << RESET << std::endl;
    }

    return num_identical_cols / num_samples;
}