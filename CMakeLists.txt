cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(centernet)

SET(CMAKE_BUILD_TYPE "Debug")

set(CMAKE_PREFIX_PATH "/data_1/everyday/0901/pytorch/torch")
find_package(Torch REQUIRED)

set(CMAKE_PREFIX_PATH "/home/yhl/software_install/opencv3.2")
find_package(OpenCV REQUIRED) #opencv里面需要有share目录才可以用find_package

#include_directories("/data_1/Anaconda1105/include")
#link_directories("/data_1/Anaconda1105/lib")
#LINK_DIRECTORIES("/data_1/Yang/software_install/Anaconda1105/lib/")

add_executable(example-app example-app.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
target_link_libraries(example-app ${OpenCV_LIBS})
set_property(TARGET example-app PROPERTY CXX_STANDARD 14)
