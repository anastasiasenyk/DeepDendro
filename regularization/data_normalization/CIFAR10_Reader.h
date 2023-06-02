//
// Created by Yaroslav Korch on 30.05.2023.
//

#ifndef DEEPDENDRO_CIFAR10_READER_H
#define DEEPDENDRO_CIFAR10_READER_H

#include <fstream>
#include <iostream>
#include <vector>
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Core>


typedef Eigen::Tensor<double, 3> Image;
typedef std::vector<Image> Images;
typedef std::vector<double> Labels;
typedef std::pair<Images, Labels> ImagesAndLabels;


ImagesAndLabels load_cifar10_batch(const std::string& filepath);

std::pair<ImagesAndLabels, ImagesAndLabels> load_cifar10_whole(const std::string& cifar_dir_bins_path);




struct CIFAR10 {
    ImagesAndLabels train;
    ImagesAndLabels test;
};

CIFAR10 read_cifar(const std::string &dir_cifar_path);


#endif //DEEPDENDRO_CIFAR10_READER_H
