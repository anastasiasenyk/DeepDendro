//
// Created by Matthew Prytula on 05.04.2023.
//

#ifndef DEEPDENDRO_MNISTPROCESS_H
#define DEEPDENDRO_MNISTPROCESS_H

#include <fstream>
#include <map>
#include "dataProcessing.h"
#include "HiddenLayer.h"

class MNISTProcess : public DataProcessing {
    int classesNum = 10;
    std::ifstream image;
    std::ifstream label;
    char number;
public:

    void skipHeaders(const std::string &imageFilename, const std::string &labelFilename,
                                   int skipBytesImg, int skipBytesLab);

    VectorXd readImg(int height, int width);

    VectorXd readLbl();

    std::map<std::string, std::pair<MatrixXd, MatrixXd>> getData(std::string pathToMNIST);

    ~MNISTProcess();
};


#endif //DEEPDENDRO_MNISTPROCESS_H
