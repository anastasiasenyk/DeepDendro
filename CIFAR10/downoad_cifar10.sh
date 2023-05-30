#!/bin/bash

target_directory="CIFAR10"

# Check if the dataset is already downloaded
if [ ! -f "$target_directory/cifar-10-batches-bin/data_batch_1.bin" ]; then
    # Check if the archive is already downloaded
    if [ ! -f "$target_directory/cifar-10-binary.tar.gz" ]; then
        curl -O https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
        mv cifar-10-binary.tar.gz "$target_directory/cifar-10-binary.tar.gz"
    fi

    # Extract the files and move them to the target directory
    tar xf "$target_directory/cifar-10-binary.tar.gz" -C "$target_directory"
    rm "$target_directory/cifar-10-binary.tar.gz"
fi
