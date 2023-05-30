cmake_minimum_required(VERSION 3.23)
project(DeepDendro)

set(CMAKE_CXX_STANDARD 20)

include_directories(.)
include_directories(/usr/local/include/eigen3)
include_directories(/usr/local/include/)


add_executable(DeepDendro
        evaluation_metrics/accuracy.cpp
        layers/hidden_layer/HiddenLayer.cpp
        layers/conv_layer/Convolutions.cpp
        layers/conv_layer/Pooling.cpp
        loss_functions/lossFunc.cpp

        model/Model.cpp
        model_management/recreate_model/recreate_model.cpp
        model_management/save_model_state/save_model.cpp

        regularization/data_normalization/dataProcessing.cpp
        regularization/data_normalization/CIFAR10_Reader.cpp
        regularization/dropout/dropout.cpp

        parallelism/inter_model/inter_model.cpp

        main.cpp activation_funcs/activationDerivative.h regularization/data_normalization/MNISTProcess.cpp regularization/data_normalization/MNISTProcess.h)


include_directories(.
        activation_funcs
        evaluation_metrics
        layers
        layers/parent_layer
        layers/hidden_layer
        layers/conv_layer
        logging

        loss_functions
        model
        model_management/recreate_model
        model_management/save_model_state
        regularization/data_normalization
        regularization/dropout
        parallelism/inter_model
        layers/flattening_layer
        )



option(DEBUG "Enable debug output" ON)
if (DEBUG)
    add_compile_definitions(DEBUG)
endif()

