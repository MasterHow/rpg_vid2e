cmake_minimum_required(VERSION 3.3)
project(esim_py)

set(OpenCV_DIR D:/TOOLKITS/OpenCV/4.5.3/opencv/build)
set(Boost_ADDITIONAL_VERSIONS "1.78" "1.78.0")
set(BOOST_ROOT D:/TOOLKITS/BOOST/boost_1_78_0)
set(Boost_FILESYSTEM_LIBRARY D:/TOOLKITS/BOOST/boost_1_78_0/stage/lib)

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

find_package(pybind11 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Boost COMPONENTS system filesystem REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 11)

include_directories(include ${EIGEN3_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})

add_library(libesim STATIC src/esim.cpp)

pybind11_add_module(esim_py src/bindings.cpp)

target_link_libraries(esim_py PRIVATE libesim ${OpenCV_LIBS} ${Boost_FILESYSTEM_LIBRARY} ${BOOST_SYSTEM_LIBRARY} Boost::filesystem Boost::system Eigen3::Eigen pybind11::embed)
