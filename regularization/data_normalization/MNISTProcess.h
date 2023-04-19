//
// Created by Matthew Prytula on 05.04.2023.
//

#ifndef DEEPDENDRO_MNISTPROCESS_H
#define DEEPDENDRO_MNISTPROCESS_H

#include <fstream>
#include <map>
#include "dataProcessing.h"
#include "HiddenLayer.h"
#include "DataSets.h"
#include <tbb/concurrent_queue.h>

class MNISTProcess : public DataProcessing {
    int classesNum = 10;
    std::ifstream image;
    std::ifstream label;
    char number;
    int numTrainImg = 60000;
    int numTestImg = 10000;
public:

    void skipHeaders(const std::string &imageFilename, const std::string &labelFilename,
                                   int skipBytesImg, int skipBytesLab);

    VectorXd readImg(int height, int width);

    VectorXd readLbl();

    DataSets getData(std::string& pathToMNIST);

    void enqueueMiniBatches(int batchSize, tbb::concurrent_queue<std::pair<MatrixXd, MatrixXd>> &queue, std::string& pathToMNIST);

    ~MNISTProcess();
};


#endif //DEEPDENDRO_MNISTPROCESS_H
