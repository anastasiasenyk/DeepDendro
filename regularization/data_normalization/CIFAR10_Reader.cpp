//
// Created by Yaroslav Korch on 30.05.2023.
//

#include "CIFAR10_Reader.h"

ImagesAndLabels load_cifar10(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if(file.is_open()) {
        std::vector<char> buffer(std::istreambuf_iterator<char>(file), {});

        Images data_vector;
        Labels labels_vector;

        for(int img = 0; img < 10000; ++img) {
            Image data_tensor(3, 32, 32);
            int offset = img * 3073; // start from label byte
            labels_vector.push_back(static_cast<unsigned char>(buffer[offset++]));
            for(int ch = 0; ch < 3; ++ch) {
                for(int px = 0; px < 32*32; ++px) {
                    data_tensor(ch, px / 32, px % 32) = static_cast<unsigned char>(buffer[offset++]);
                }
            }
            data_vector.push_back(data_tensor);
        }
        file.close();
        return {data_vector, labels_vector};
    } else {
        std::cerr << "Could not open the file!" << std::endl;
        exit(1);
    }
}


std::pair<ImagesAndLabels, ImagesAndLabels> load_cifar10_whole(const std::string& cifar_dir_bins_path){
    std::array<std::string, 6> bins_paths = {
            cifar_dir_bins_path + "/data_batch_1.bin",
            cifar_dir_bins_path + "/data_batch_2.bin",
            cifar_dir_bins_path + "/data_batch_3.bin",
            cifar_dir_bins_path + "/data_batch_4.bin",
            cifar_dir_bins_path + "/data_batch_5.bin",
            cifar_dir_bins_path + "/test_batch.bin"
    };

    ImagesAndLabels training_vector;

    for(size_t i = 0; i < 4; ++i) {
        ImagesAndLabels data = load_cifar10(bins_paths[i]);
        training_vector.first.insert(training_vector.first.end(), data.first.begin(), data.first.end());
        training_vector.second.insert(training_vector.second.end(), data.second.begin(), data.second.end());
    }

    ImagesAndLabels test_vector = load_cifar10(bins_paths[5]);

    return {training_vector, test_vector};
}