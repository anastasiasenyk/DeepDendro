cmake_minimum_required(VERSION 3.24)
project(DeepDendro)

set(CMAKE_CXX_STANDARD 17)

include_directories(.)

add_executable(DeepDendro
        "layers/parent_layer/Layer.cpp"
        )

include_directories(. "layers/parent_layer/Layer.h")
