cmake_minimum_required(VERSION 3.24)
project(DeepDendro)

set(CMAKE_CXX_STANDARD 20)

include_directories(.)
include_directories(/usr/local/include/eigen3)



add_executable(DeepDendro
        activation_funcs/activation_funcs.cpp
        evaluation_metrics/accuracy.cpp
        layers/hidden_layer/HiddenLayer.cpp
        logging/Logger.cpp
        loss_functions/basic_loss_func.cpp

        model/Model.cpp
        model_management/recreate_model/recreate_model.cpp
        model_management/save_model_state/save_model.cpp
        model/to_predict/predict.cpp

        regularization/data_normalization/data_norm.cpp
        regularization/dropout/dropout.cpp

        main.cpp)


include_directories(.
        activation_funcs
        evaluation_metrics
        layers
        layers/parent_layer
        layers/hidden_layer
        logging
        loss_functions
        matrices
        model
        model_management/recreate_model
        model_management/save_model_state
        model/to_predict
        regularization/data_normalization
        regularization/dropout
        )



option(DEBUG "Enable debug output" ON)
if (DEBUG)
    add_compile_definitions(DEBUG)
endif()


