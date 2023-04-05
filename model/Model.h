//
// Created by Yaroslav Korch on 30.03.2023.
//

#ifndef DEEPDENDRO_MODEL_H
#define DEEPDENDRO_MODEL_H

#include "Layers.h"
#include "vector"
#include "activationFuncs.h"
#include "lossFunc.h"
#include "logging.h"


class Model {
    std::vector<HiddenLayer> layers;
    MatrixXd train_data;
    MatrixXd train_labels;
    HiddenLayer *save_prev_layer;

public:
    Model();

    void addInput(const MatrixXd &data);

    void addOutput(const MatrixXd &labels);

    // by default, we have a straight-forward model (no branching)
    void addLayer(int neurons, activation activationType);


    void train(size_t epochs = 10, double learning_rate = 0.005);

    void test();

    void create_mini_batches();

    MatrixXd predict_after_forward_prop();

    MatrixXd predict(const MatrixXd &data);

    double calc_accuracy(const MatrixXd &predicted, const MatrixXd &true_labels);
};


#endif //DEEPDENDRO_MODEL_H
