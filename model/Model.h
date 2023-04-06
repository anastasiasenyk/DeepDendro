//
// Created by Yaroslav Korch on 30.03.2023.
//

#ifndef DEEPDENDRO_MODEL_H
#define DEEPDENDRO_MODEL_H
#define YELLOW  "\033[33m"
#define RESET   "\033[0m"

#include <iostream>
#include "Layers.h"
#include "vector"
#include "activationFuncs.h"
#include "lossFunc.h"
#include "logging.h"


class Model {
    std::vector<HiddenLayer> layers;
    MatrixXd train_data;
    MatrixXd train_labels;
    std::shared_ptr<HiddenLayer> save_prev_layer;

    MatrixXd predict_after_forward_prop();
    void create_mini_batches();

public:
    Model();

    void addInput(const MatrixXd &data);

    void addOutput(const MatrixXd &labels);

    // by default, we have a straight-forward model (no branching)
    void addLayer(int neurons, activation activationType);

    void train(size_t epochs = 10, double learning_rate = 0.005);

    MatrixXd predict(const MatrixXd &testData);
    double calc_accuracy(const MatrixXd &predicted, const MatrixXd &true_labels, bool verbose = false);

    void test();
};


#endif //DEEPDENDRO_MODEL_H
