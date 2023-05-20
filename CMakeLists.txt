cmake_minimum_required(VERSION 3.23)
project(DeepDendro)

set(CMAKE_CXX_STANDARD 20)

include_directories(.)
include_directories(/usr/local/include/eigen3)
#include_directories(/opt/homebrew/Cellar/tbb/2021.9.0/include)
#
#set(TBB_DIR /opt/homebrew/Cellar/tbb/2021.9.0/lib/cmake/TBB/)
#find_package(TBB REQUIRED)

# Set the path to the TBB include directory
set(TBB_INCLUDE_DIRS "/opt/homebrew/include")

# Set the path to the TBB library directory
set(TBB_LIBRARY_DIRS "/opt/homebrew/lib")

# Find the TBB library
find_library(TBB_LIBRARIES tbb HINTS ${TBB_LIBRARY_DIRS})

# Add the TBB include directory to the include path
include_directories(${TBB_INCLUDE_DIRS})

#set(CMAKE_PREFIX_PATH "/opt/local")
set(TBB_ROOT /opt/homebrew/Cellar/tbb/2021.9.0)

find_package(TBB REQUIRED)
include_directories(${TBB_INCLUDE_DIRS})
include_directories(/opt/homebrew/include)


add_executable(DeepDendro
        activation_funcs/activationFuncs.cpp
        evaluation_metrics/accuracy.cpp
        layers/hidden_layer/HiddenLayer.cpp
        loss_functions/lossFunc.cpp

        model/Model.cpp
        model_management/recreate_model/recreate_model.cpp
        model_management/save_model_state/save_model.cpp
        model/to_predict/predict.cpp

        regularization/data_normalization/dataProcessing.cpp
        regularization/dropout/dropout.cpp

        parallelism/inter_model/inter_model.cpp

        main.cpp activation_funcs/activationDerivative.cpp activation_funcs/activationDerivative.h regularization/data_normalization/MNISTProcess.cpp regularization/data_normalization/MNISTProcess.h Worker/Worker.cpp Worker/Worker.h Worker/FirstWorker.cpp Worker/FirstWorker.h Worker/LastWorker.cpp Worker/LastWorker.h layers/flow_layer/FlowLayer.cpp layers/flow_layer/FlowLayer.h layers/flow_layer/FlowOutputLayer.cpp layers/flow_layer/FlowOutputLayer.h)

#target_link_libraries(DeepDendro PRIVATE TBB::tbb)

# Link against the TBB library
#target_link_libraries(DeepDendro ${TBB_LIBRARIES})
target_include_directories(DeepDendro PRIVATE ${TBB_INCLUDE_DIRS})
target_link_libraries(DeepDendro PRIVATE ${TBB_LIBRARIES})


include_directories(.
        activation_funcs
        evaluation_metrics
        layers
        layers/parent_layer
        layers/hidden_layer
        logging

        loss_functions
        model
        model_management/recreate_model
        model_management/save_model_state
        model/to_predict
        regularization/data_normalization
        regularization/dropout
        parallelism/inter_model
        Worker
        )



option(DEBUG "Enable debug output" OFF)
if (DEBUG)
    add_compile_definitions(DEBUG)
endif()

option(LOGGING "Enable debug output" ON)
if (LOGGING)
    add_compile_definitions(LOGGING)
endif()

