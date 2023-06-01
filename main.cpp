//
// Created by Yaroslav Korch on 30.03.2023.
//

#include "Model.h"
#include "MNISTProcess.h"
#include "HiddenLayer.h"
#include <thread>
#include "PipelineModel.h"


int main() {
//    srand((unsigned int) time(0));

    std::string pathToMNIST = "../MNIST_ORG";
    MNISTProcess mnistProcessTrain = MNISTProcess();
//
//    DataSets data = mnistProcessTrain.getData(pathToMNIST);

//    tbb::concurrent_queue<std::pair<MatrixXd, MatrixXd>> miniBatchQ;
//    mnistProcessTrain.enqueueMiniBatches(32, miniBatchQ, pathToMNIST);

//    std::cout << miniBatchQ.unsafe_size() << std::endl;
    PipelineModel pipelineModel(8);
    pipelineModel.addLayer(16, {784, 8}, relu, 8);
    pipelineModel.addLayer(8, {16, 8}, relu, 8);
//    pipelineModel.run_pipeline(miniBatchQ);
    pipelineModel.runConfPipeline();

//    Model model;
//    model.addInput(data.trainData);
//    model.addOutput(data.trainLabels);
//
//    model.addLayer(16, activation::relu);
//    model.addLayer(8, activation::relu);
//
//    model.train(10, 0.05);
//    model.calc_accuracy(model.predict(data.testData), data.testLabels, true);
    return 0;
}