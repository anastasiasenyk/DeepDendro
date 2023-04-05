//
// Created by Yaroslav Korch on 30.03.2023.
//

#include "Model.h"
#include "MNISTProcess.h"

int main() {
    int numTrainImg = 60000;
    srand((unsigned int) time(0));

    MNISTProcess mnistProcess = MNISTProcess();
    mnistProcess.skipHeaders("../MNIST_ORG/train-images.idx3-ubyte", "../MNIST_ORG/train-labels.idx1-ubyte", 16, 8);
    MatrixXd images(784, numTrainImg);
    MatrixXd labels(10, numTrainImg);
    for (int sample = 0; sample < numTrainImg; ++sample) {
        VectorXd oneHotLabel = mnistProcess.readLbl();
        VectorXd flattenedImage = mnistProcess.readImg(28, 28);
        images.col(sample) = flattenedImage;
        labels.col(sample) = oneHotLabel;
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
    model.addInput(images);
    model.addOutput(labels);

    model.addLayer(16, activation::relu);
    model.addLayer(8, activation::relu);
    model.train(10, 0.005);

    return 0;
}