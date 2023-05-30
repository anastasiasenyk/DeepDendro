//
// Created by Matthew Prytula on 30.05.2023.
//

#ifndef DEEPDENDRO_MICROBATCHSOURCEBODY_H
#define DEEPDENDRO_MICROBATCHSOURCEBODY_H
#include <tbb/concurrent_queue.h>
#include <Eigen/Dense>

using Eigen::MatrixXd;

class MicrobatchSourceBody {
    tbb::concurrent_queue<std::pair<MatrixXd, MatrixXd>>& minibatchQueue;
    std::pair<MatrixXd, MatrixXd> currentMinibatch;
    int currentMicrobatchStart;
    int microbatchSize;

public:
    MicrobatchSourceBody(tbb::concurrent_queue<std::pair<MatrixXd, MatrixXd>>& minibatchQueue, int microbatchSize);

    bool operator()(std::pair<MatrixXd, MatrixXd>& outputMicrobatch);
};


#endif //DEEPDENDRO_MICROBATCHSOURCEBODY_H
