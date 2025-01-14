
## Requirements

- Python 3.7
- C++ compiler (eg. Windows: Visual Studio 15 2017, Ubuntu: g++)
- pybind11 `https://github.com/pybind/pybind11.git`
- Linear Assignment Problem solver `https://github.com/gatagat/lap.git`
## Install

Install Eigen for Windows (after the following steps, add include directory `C:\eigen-3.4.0` for example.)
1) Download Eigen 3.4.0 (NOT lower than this version) from it official website https://eigen.tuxfamily.org/ or [ZIP file here](https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip).
2) `mkdir build_dir`
3) `cd build_dir`
4) `cmake ../`
5) `make install`, this step does not require

Install Eigen for Linux
1) [install and use eigen3 on ubuntu 16.04](https://kezunlin.me/post/d97b21ee/) 
2) `sudo apt-get install libeigen3-dev` libeigen3-dev is installed install to `/usr/include/eigen3/` and `/usr/lib/cmake/eigen3`.
3) We must make a change in [CMakeLists.txt](CMakeLists.txt) `SET( EIGEN3_INCLUDE_DIR "/usr/local/include/eigen3" )` to `SET( EIGEN3_INCLUDE_DIR "/usr/include/eigen3/" )`.

Include OpenCV headers
1) Install OpenCV `sudo apt install libopencv-dev`
2) Find where OpenCV is instsallted: `pkg-config --cflags opencv4`
3) Change directory in [CMakeLists.txt](CMakeLists.txt) if needed `include_directories(/usr/include/opencv4/)`

Build
1) Build C++ Package to call from Python: `python setup.py build develop`
2) Build to run directly from C++:  
   you need to change path of detect files and images in [`main.cpp`](src%2Fmain.cpp)    
   Comment all lines related to `pybind11` in [`CMakeLists.txt`](CMakeLists.txt) and [`main.cpp`](src%2Fmain.cpp), Uncomment `add_executable` in [`CMakeLists.txt`](CMakeLists.txt)     
   Build: `mkdir build` > `cd build` > `cmake ..` > `cmake --build .` > and run it by `./adaptbytetrack`

LICENCE
=======
Linh Ma, Machine Learning & Vision Laboratory, GIST, South Korea