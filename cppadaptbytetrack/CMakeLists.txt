cmake_minimum_required(VERSION 3.14)
project(adaptbytetrack)

#find_package (Eigen3 3.4 REQUIRED NO_MODULE)
SET(EIGEN3_INCLUDE_DIR "/media/ubuntu/2715608D71CBF6FC/eigen-3.4.0")
IF (NOT EIGEN3_INCLUDE_DIR)
    MESSAGE(FATAL_ERROR "Please point the environment variable EIGEN3_INCLUDE_DIR to the include directory of your Eigen3 installation.")
ENDIF ()
INCLUDE_DIRECTORIES("${EIGEN3_INCLUDE_DIR}")

# Include OpenCV headers
# include_directories(${OpenCV_INCLUDE_DIRS})
# sudo apt install libopencv-dev, pkg-config --cflags opencv4
include_directories(/usr/include/opencv4/)


#if (MSVC)
#    add_compile_options(/bigobj)
#else ()
#    add_compile_options(-Wa,-mbig-obj)
#endif ()

##### OpenMP
# find_package(OpenMP)
#  if (OPENMP_FOUND)
#      set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#      set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#  endif()

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(pybind11)
pybind11_add_module(adaptbytetrack
    src/main.cpp
    src/kalmanFilter.cpp
    src/lapjv.cpp
    src/STrack.cpp
    src/utils.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
# Link OpenCV libraries (reading and displaying images)
target_link_libraries(${PROJECT_NAME} PRIVATE opencv_core opencv_imgproc opencv_highgui)
# Link OpenCV libraries (OpenCV GlobalMotionCompensation)
target_link_libraries(${PROJECT_NAME} PRIVATE opencv_imgcodecs opencv_features2d opencv_video opencv_calib3d opencv_videostab)


# Display OpenCV info
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")