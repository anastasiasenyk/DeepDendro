//
// Created by Matthew Prytula on 05.04.2023.
//

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



MNISTProcess::~MNISTProcess() {
    image.close();
}