#include <filesystem>
#include <iostream>
#include <fstream>
#include<random>
#include "BYTETracker.cpp"
#include "GlobalMotionCompensation.cpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

PYBIND11_MODULE(adaptbytetrack, m) {
    // Adaptive ByteTracker
    py::class_<BYTETracker>(m, "BYTETracker")
            .def(py::init<int, int, float, float, bool > ())
            .def("update", &BYTETracker::update);
}

using namespace std;
using namespace Eigen;

MatrixXd read_txt(string file_name = "", char delim = ' ') {
    std::ifstream inRowCol(file_name);
    std::string lineRowCol;
    int nrows = 0;
    int ncols = 0;
    while (std::getline(inRowCol, lineRowCol)) {
        nrows++;
        // Split the line by comma and count the number of elements
        std::vector<std::string> res;
        std::stringstream ss(lineRowCol);
        std::string element;
        ncols = 0;
        while (std::getline(ss, element, delim)) {
            ncols++;
        }
    }
    if (ncols == 0) {
        return MatrixXd(nrows, ncols); // column vector, no delimiter
    }

    std::ifstream in(file_name);
    std::string line;
    int row = 0;
    int col = 0;
    Eigen::MatrixXd res = Eigen::MatrixXd(nrows, ncols);
    if (in.is_open()) {
        while (std::getline(in, line)) {
            char *ptr = (char *) line.c_str();
            int len = line.length();
            col = 0;
            char *start = ptr;
            for (int i = 0; i < len; i++) {
                if (ptr[i] == delim) {
                    res(row, col++) = atof(start);
                    start = ptr + i + 1;
                }
            }
            res(row, col) = atof(start);
            row++;
        }
        in.close();
    }
    return res;
}

cv::Scalar get_color(int idx) {
    idx = (idx + 1) * 50;
    int r = (37 * idx) % 255;
    int g = (17 * idx) % 255;
    int b = (29 * idx) % 255;

    return cv::Scalar(b, g, r);  // OpenCV uses BGR color format
}

//int main() {
//
//
//
//
//    MatrixXd arr = read_txt("../../detection/bytetrack/MOT16-02.txt", ',');
//    int n_frames = arr.col(0).maxCoeff();
//
//    BYTETracker byteTracker(30, 30);
//    vector<cv::String> fn;
//    glob("/media/ubuntu/2715608D71CBF6FC/datasets/mot/MOT16/train/MOT16-02/img1/*.jpg", fn, false);
//
//    for (int frame = 0; frame < n_frames; frame++) {
//        std::vector<vector<float>> rects;
//        for (int i = 0; i < arr.rows(); i++) {
//            if (arr(i, 0) == frame) {
//                float left = static_cast<float>(arr(i, 1));
//                float top = static_cast<float>(arr(i, 2));
//                float right = static_cast<float>(arr(i, 3));
//                float bottom = static_cast<float>(arr(i, 4));
//                float conf = static_cast<float>(arr(i, 5));
//                vector<float> row = {left, top, right, bottom, conf};
//                rects.emplace_back(row);
//            }
//        }
//
//        cv::Mat img = cv::imread(fn[frame]);
//        vector<vector<float>> output_tracks = byteTracker.update(rects, img);
//
//        cv::putText(img, //target image
//                    "Frame " + std::to_string(frame), //text
//                    cv::Point(30, 30), //top-left position
//                    cv::FONT_HERSHEY_DUPLEX,
//                    1,
//                    CV_RGB(255, 0, 0), //font color
//                    2);
//        for (vector<float> track: output_tracks) {
//            Scalar c = get_color((int) track[0]);
//            cv::Rect rect(track[1], track[2], track[3], track[4]);
//            cv::rectangle(img, rect, c, 2);
//            cv::putText(img, //target image
//                        std::to_string(int(track[0])), //text
//                        cv::Point(track[1] + 5, track[2] + 20), //top-left position
//                        cv::FONT_HERSHEY_DUPLEX,
//                        0.8,
//                        c, //font color
//                        1);
//        }
//        cv::imshow("DKM", img);
//        cv::waitKey(10);
//    }
//
//
//    return 0;
//}