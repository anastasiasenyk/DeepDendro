//
// Created by Matthew Prytula on 30.05.2023.
//

#include "MicrobatchSourceBody.h"

MicrobatchSourceBody::MicrobatchSourceBody(tbb::concurrent_queue<std::pair<MatrixXd, MatrixXd>> &minibatchQueue,
                                           int microbatchSize)
        : minibatchQueue(minibatchQueue), currentMicrobatchStart(0), microbatchSize(microbatchSize) {}

bool MicrobatchSourceBody::operator()(std::pair<MatrixXd, MatrixXd> &outputMicrobatch) {
    if (currentMicrobatchStart >= currentMinibatch.first.cols()) {
        // Fetch a new minibatch
        if (!minibatchQueue.try_pop(currentMinibatch)) {
            return false;
        }
        currentMicrobatchStart = 0;
    }

    int microbatchEnd = std::min(currentMicrobatchStart + microbatchSize, (int)currentMinibatch.first.cols());

    // Slice the minibatch into a microbatch
    outputMicrobatch.first = currentMinibatch.first.middleCols(currentMicrobatchStart, microbatchEnd - currentMicrobatchStart);
    outputMicrobatch.second = currentMinibatch.second.middleCols(currentMicrobatchStart, microbatchEnd - currentMicrobatchStart);

    currentMicrobatchStart = microbatchEnd;

    return true;
}
