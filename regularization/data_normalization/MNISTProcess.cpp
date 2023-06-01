//
// Created by Matthew Prytula on 05.04.2023.
//

#include "Layers.h"
#include "MNISTProcess.h"


void MNISTProcess::skipHeaders(const std::string &imageFilename, const std::string &labelFilename,
                               int skipBytesImg, int skipBytesLab) {
    image.open(imageFilename.c_str(), std::ios::in | std::ios::binary);
    label.open(labelFilename.c_str(), std::ios::in | std::ios::binary);
    if (!image) {
        std::cerr << "Failed to open image file: " << imageFilename << std::endl;
        return;
    }
    if(!label) {
        std::cerr << "Failed to open label file: " << imageFilename << std::endl;
        return;
    }
    // skip header
    for (int i = 1; i <= skipBytesImg; ++i) {
        image.read(&number, sizeof(char));
    }
    for(int i=1; i<= skipBytesLab; ++i) {
        label.read(&number, sizeof(char));
    }
}

VectorXd MNISTProcess::readImg(int height, int width) {
    MatrixXd imgMatrix(height, width);  // create a MatrixXd to store the image data
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            image.read(&number, sizeof(char));
            imgMatrix(j, i) = (number==0)? 0: 1;
        }
    }
    return VectorXd::Map(imgMatrix.data(), height * width);;
}

VectorXd MNISTProcess::readLbl() {
    label.read(&number, sizeof(char));
    MatrixXd oneHot(classesNum, 1);
    for (int i = 0; i < classesNum; ++i) {
        oneHot(i, 0) = 0.0;
    }
    oneHot(number, 0) = 1.0;
    return oneHot;
}

void MNISTProcess::enqueueMiniBatches(int batchSize, tbb::concurrent_queue<std::pair<MatrixXd, MatrixXd>> &queue, std::string& pathToMNIST) {

    skipHeaders(pathToMNIST + "/train-images.idx3-ubyte", pathToMNIST + "/train-labels.idx1-ubyte", 16, 8);

    for (int sample = 0; sample < numTrainImg; sample += batchSize) {
        MatrixXd batchData(784, std::min(batchSize, numTrainImg - sample));
        MatrixXd batchLabels(10, std::min(batchSize, numTrainImg - sample));

        // Read the images and labels for this batch
        for (int i = 0; i < std::min(batchSize, numTrainImg - sample); ++i) {
            VectorXd oneHotLabel = readLbl();
            VectorXd flattenedImage = readImg(28, 28);

            batchData.col(i) = flattenedImage;
            batchLabels.col(i) = oneHotLabel;

        }

        // Enqueue the mini-batch
        queue.push(std::make_pair(batchData, batchLabels));
    }
    queue.push(std::make_pair(MatrixXd::Constant(1, 1, -1.0), MatrixXd::Constant(1, 1, -1.0)));
}

TestSet MNISTProcess::getTestingData(std::string& pathToMNIST) {
    TestSet data = TestSet();
    MNISTProcess mnistProcessTest = MNISTProcess();

    mnistProcessTest.skipHeaders(pathToMNIST + "/t10k-images.idx3-ubyte", pathToMNIST + "/t10k-labels.idx1-ubyte", 16, 8);

    data.testData = MatrixXd (784, numTestImg);
    data.testLabels = MatrixXd (10, numTestImg);

    for (int sample = 0; sample < numTestImg; ++sample) {
        VectorXd oneHotLabel = mnistProcessTest.readLbl();
        VectorXd flattenedImage = mnistProcessTest.readImg(28, 28);

        data.testData.col(sample) = flattenedImage;
        data.testLabels.col(sample) = oneHotLabel;
    }

    return data;
}


TrainingSet MNISTProcess::getTrainingData(std::string& pathToMNIST) {

        TrainingSet data = TrainingSet();

        MNISTProcess mnistProcessTrain = MNISTProcess();
        mnistProcessTrain.skipHeaders(pathToMNIST + "/train-images.idx3-ubyte", pathToMNIST + "/train-labels.idx1-ubyte", 16, 8);

        data.trainData = MatrixXd (784, numTrainImg);
        data.trainLabels = MatrixXd (10, numTrainImg);

        for (int sample = 0; sample < numTrainImg; ++sample) {
            VectorXd oneHotLabel = mnistProcessTrain.readLbl();
            VectorXd flattenedImage = mnistProcessTrain.readImg(28, 28);

            data.trainData.col(sample) = flattenedImage;
            data.trainLabels.col(sample) = oneHotLabel;
        }

        return data;
}

void MNISTProcess::enqueueMiniBatchesFromMemory(int batchSize, tbb::concurrent_queue<std::pair<MatrixXd, MatrixXd>> &queue, MatrixXd& trainData, MatrixXd& trainLabels) {
    for (int sample = 0; sample < trainData.cols(); sample += batchSize) {
        int currentBatchSize = std::min(batchSize, (int)trainData.cols() - sample);

        MatrixXd batchData = trainData.middleCols(sample, currentBatchSize);
        MatrixXd batchLabels = trainLabels.middleCols(sample, currentBatchSize);

        // Enqueue the mini-batch
        queue.push(std::make_pair(batchData, batchLabels));
    }
    queue.push(std::make_pair(MatrixXd::Constant(1, 1, -1.0), MatrixXd::Constant(1, 1, -1.0)));
}



DataSets MNISTProcess::getData(std::string& pathToMNIST) {

    DataSets data = DataSets();

    MNISTProcess mnistProcessTrain = MNISTProcess();
    mnistProcessTrain.skipHeaders(pathToMNIST + "/train-images.idx3-ubyte", pathToMNIST + "/train-labels.idx1-ubyte", 16, 8);
    MNISTProcess mnistProcessTest = MNISTProcess();
    mnistProcessTest.skipHeaders(pathToMNIST + "/t10k-images.idx3-ubyte", pathToMNIST + "/t10k-labels.idx1-ubyte", 16, 8);

    data.trainData = MatrixXd (784, numTrainImg);
    data.trainLabels = MatrixXd (10, numTrainImg);
    data.testData = MatrixXd (784, numTestImg);
    data.testLabels = MatrixXd (10, numTestImg);


    for (int sample = 0; sample < numTrainImg; ++sample) {
        VectorXd oneHotLabel = mnistProcessTrain.readLbl();
        VectorXd flattenedImage = mnistProcessTrain.readImg(28, 28);
        data.trainData.col(sample) = flattenedImage;
        data.trainLabels.col(sample) = oneHotLabel;
    }

    for (int sample = 0; sample < numTestImg; ++sample) {
        VectorXd oneHotLabel = mnistProcessTest.readLbl();
        VectorXd flattenedImage = mnistProcessTest.readImg(28, 28);
        data.testData.col(sample) = flattenedImage;
        data.testLabels.col(sample) = oneHotLabel;
    }
    return data;
}

void MNISTProcess::reset() {
    image.close();
    label.close();
}

MNISTProcess::~MNISTProcess() {
    reset();
}