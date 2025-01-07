#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videostab.hpp>
#include <opencv2/videostab/global_motion.hpp>
#include <Eigen/Core>

// Affine Motion Estimation or Affine Global Motion Estimation using Sparse Optical Flow
class GlobalMotionCompensation {
private:
    cv::Mat _prev_frame;
    std::vector<cv::KeyPoint> _prev_keypoints;
    bool _first_frame_initialized = false;
    float _downscale = 1.0;
public:
    GlobalMotionCompensation() = default;

    ~GlobalMotionCompensation();

    Eigen::MatrixXf apply(const cv::Mat &frame_raw);
};