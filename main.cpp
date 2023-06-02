//
// Created by Yaroslav Korch on 30.03.2023.
//

#include "Model.h"
#include "MNISTProcess.h"
#include "Filter.h"
#include "Pooling.h"
#include "Convolutions.h"
#include "FlatteningLayer.h"
#include "CIFAR10_Reader.h"

using std::cout;



int main() {
    srand((unsigned int) time(0));




    MNISTProcess mnistProcessTrain = MNISTProcess();
    DataSets data = mnistProcessTrain.getData("../MNIST_ORG");

    size_t N_FILTERS = 4;
    Convolutional2D conv1{
            N_FILTERS,
            {5, 5},
            activation::none,
            {28, 28}
    };

    // 4608
    FlatteningLayer2D flat;

    HiddenLayer inBetween{32, {24 * 24 * N_FILTERS, 1}, activation::relu};

    HiddenLayer denseFinal{
            10,
            {32, 1},
            activation::softmax
    };

    Eigen::VectorXd vec_flat(24 * 24 * N_FILTERS);
    Eigen::MatrixXd mat_flat(24 * 24 * N_FILTERS, 10);
    Eigen::MatrixXd labels(10, 10);

    Eigen::Tensor<double, 3> delta(10, 24, (Eigen::Index) (24 * N_FILTERS));
    labels.setZero();

    const auto cycle_one_batch = [&](const Eigen::Tensor<double, 3> &images, const Eigen::MatrixXd &correct_labels,
                                     bool to_kill = false) {


        auto conv1_out = conv1.forward_prop(images);


        for (int i = 0; i < 10; ++i) {
            Eigen::Tensor<double, 2> part = conv1_out.chip(i, 0);
            vec_flat = flat.flatten(part);
            mat_flat.col(i) = vec_flat;
        }


        inBetween.forward_prop(mat_flat);


        auto to_come = inBetween.getAValues();


        denseFinal.forward_prop(to_come);


        auto almost_dense = denseFinal.calc_first_back_prop(correct_labels);
        auto out_between = inBetween.calc_back_prop(almost_dense);


//        std::cout << "grad: " << out_between.sum() << "\n";
//        if (out_between.sum() == 0){
//            exit(0);
//        }

        denseFinal.apply_back_prop(0.005, to_come);
        inBetween.apply_back_prop(0.005, mat_flat);

        for (int i = 0; i < 10; ++i) {
            delta.chip(i, 0) = flat.back_to_tensor(out_between.col(i));
        }

        auto not_needed_grad = conv1.calc_back_prop(delta);

        conv1.apply_back_prop(0.005);
    };

    Eigen::MatrixXd mat_test(24 * 24 * N_FILTERS, 1);
    const auto predict_one_image = [&](const Eigen::Tensor<double, 3> &images) {

        auto conv1_out = conv1.forward_prop(images);


        Eigen::Tensor<double, 2> part = conv1_out.chip(0, 0);
        vec_flat = flat.flatten(part);
        mat_test.col(0) = vec_flat;

        inBetween.forward_prop(mat_test);
        denseFinal.forward_prop(inBetween.getAValues());

        return denseFinal.getAValues();
    };


    const size_t epochs = 3;


    Eigen::Tensor<double, 3> images_batch(10, 28, 28);


    for (size_t i = 0; i < epochs; ++i) {
        for (size_t j = 0; j < 6000; ++j) {

            for (int k = 0; k < 10; ++k) {
                images_batch.chip(k, 0) = data.trainData.chip(j * 10 + k, 0) - 0.5;
            }
            Eigen::MatrixXd subMatrix = data.trainLabels.block(0, j * 10, data.trainLabels.rows(), 10);
            cycle_one_batch(images_batch, subMatrix);

        }

        cout << "Epoch " << i + 1 << " finished!!!\n";
    }


    Eigen::Tensor<double, 3> images_batch_test(1, 28, 28);

    const size_t test_size = 9000;
    size_t correct = 0;
    Eigen::Index max_index;

    for (size_t i = 0; i < test_size; ++i) {
        images_batch_test.chip(0, 0) = data.testData.chip(i, 0) - 0.5;
        MatrixXd correct_label = data.testLabels.block(0, i, data.testLabels.rows(), 1);

        auto prediction = predict_one_image(images_batch_test);
        auto max_val = prediction.maxCoeff();

        for (Eigen::Index h = 0; h < 10; ++h) {
            if (prediction(h) == max_val) {
                max_index = h;
                break;
            }
        }
        correct += (correct_label(max_index) == 1);
    }


    cout << "\nCorrect: " << correct << "\n";
    cout << "Total: " << test_size << "\n";

    cout << "Accuracy after training: " << correct / ((double) test_size) << "\n";


    exit(0);


}