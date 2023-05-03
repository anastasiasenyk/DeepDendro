//
// Created by Yaroslav Korch on 30.03.2023.
//

#include "Model.h"
#include "MNISTProcess.h"
#include "inter_model.h"
#include "FirstWorker.h"
#include "Worker.h"
#include "LastWorker.h"
#include "HiddenLayer.h"
#include <tbb/parallel_invoke.h>


int main() {
//    srand((unsigned int) time(0));

    std::string pathToMNIST = "../MNIST_ORG";
    MNISTProcess mnistProcessTrain = MNISTProcess();
//
//    DataSets data = mnistProcessTrain.getData(pathToMNIST);


    HiddenLayer hiddenLayer1(16, {784, 32}, find_activation_func(activation::relu));
    HiddenLayer hiddenLayer2(8, hiddenLayer1.shape, find_activation_func(activation::relu));
    HiddenLayer hiddenLayer3(10, hiddenLayer2.shape, find_activation_func(activation::softmax));




    FirstWorker worker1(1, hiddenLayer1);
    Worker worker2(2, hiddenLayer2, 7500);
    LastWorker worker3(3, hiddenLayer3, 7500);
    tbb::task_group tg;
    for (int i = 0; i < 10; ++i) {
        tbb::concurrent_queue<std::pair<MatrixXd, MatrixXd>> mainQ;
        mnistProcessTrain.enqueueMiniBatches(32, mainQ, pathToMNIST);
        tg.run([&] { worker1(8, mainQ, worker2.activationsQ, 25, 0.05); });
        tg.run([&] { worker2(worker3.activationsQ, worker1.deltaWQ, worker1.gradients_pushed, 25, 0.05); });
        tg.run([&] { worker3(worker2.deltaWQ, worker2.gradients_pushed, 0.05); });
        tg.wait();
        std::cout << "======================\n";
        mnistProcessTrain.reset();
    }





//    Model model;
//    model.addInput(data.trainData);
//    model.addOutput(data.trainLabels);
//
//    model.addLayer(16, activation::relu);
//    model.addLayer(8, activation::relu);
//
//    model.train(100, 0.05);
//    model.calc_accuracy(model.predict(data.testData), data.testLabels, true);
    return 0;
}