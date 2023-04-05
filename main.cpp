//
// Created by Yaroslav Korch on 30.03.2023.
//

#include "Model.h"
#include "MNISTProcess.h"


int main() {
    srand((unsigned int) time(0));

    int numTrainImg = 60000;
    int numTestImg = 10000;


    MNISTProcess mnistProcessTrain = MNISTProcess();
    mnistProcessTrain.skipHeaders("../MNIST_ORG/train-images.idx3-ubyte", "../MNIST_ORG/train-labels.idx1-ubyte", 16, 8);
    MNISTProcess mnistProcessTest = MNISTProcess();
    mnistProcessTest.skipHeaders("../MNIST_ORG/t10k-images.idx3-ubyte", "../MNIST_ORG/t10k-labels.idx1-ubyte", 16, 8);

    MatrixXd trainImages(784, numTrainImg);
    MatrixXd trainLabels(10, numTrainImg);
    MatrixXd testImages(784, numTestImg);
    MatrixXd testLabels(10, numTestImg);



    for (int sample = 0; sample < numTrainImg; ++sample) {
        VectorXd oneHotLabel = mnistProcessTrain.readLbl();
        VectorXd flattenedImage = mnistProcessTrain.readImg(28, 28);
        trainImages.col(sample) = flattenedImage;
        trainLabels.col(sample) = oneHotLabel;
    }

    for (int sample = 0; sample < numTestImg; ++sample) {
        VectorXd oneHotLabel = mnistProcessTest.readLbl();
        VectorXd flattenedImage = mnistProcessTest.readImg(28, 28);
        testImages.col(sample) = flattenedImage;
        testLabels.col(sample) = oneHotLabel;
    }

#ifdef DEBUG
    for(int k=0;k<10;++k) {
        for (int i = 0; i < 10; ++i) {
            std::cout << labels(i, k) << std::endl;
        }

        for (int i = 0; i < 28; ++i) {
            for (int j = 0; j < 28; ++j) {
                std::cout << images(j*28+i, k) << "";
            }
            std::cout << "\n";
        }
    }
#endif


    Model model;
    model.addInput(trainImages);
    model.addOutput(trainLabels);

    model.addLayer(16, activation::relu);
    model.addLayer(16, activation::relu);
    model.train(10, 0.005);

    return 0;
}