cmake_minimum_required(VERSION 3.22)
project(LaneDetection VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED HINTS ${CMAKE_SOURCE_DIR}/thirdparty/OpenCV/install/lib/cmake/opencv4)

if(OpenCV_FOUND)
    message(STATUS "OpenCV Found! - ${OpenCV_DIR}")
endif()

include_directories(LaneDetection PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    include
)

add_executable(LaneDetection src/main.cpp)

target_link_libraries(LaneDetection PRIVATE
    ${OpenCV_LIBS}
)

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
