#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videostab.hpp>
#include <opencv2/videostab/global_motion.hpp>
#include <Eigen/Core>

/**
 * @brief A class for performing global motion compensation on video frames. Affine Motion Estimation or Affine Global Motion Estimation using Sparse Optical Flow
 *
 * This class is designed to estimate and compensate for global motion (e.g., camera motion) between consecutive video frames.
 * It tracks keypoints from a previous frame and applies motion compensation to align the current frame, useful in video stabilization
 * or tracking applications.
 */
class GlobalMotionCompensation {
private:
    cv::Mat _prev_frame; // Stores the previous video frame for motion estimation.
    std::vector <cv::KeyPoint> _prev_keypoints; // Stores keypoints detected in the previous frame.
    bool _first_frame_initialized = false; // Indicates whether the first frame has been initialized.
    float _downscale = 1.0; // Default value indicating no downscaling, preserving original frame resolution for full-detail processing.
public:
    GlobalMotionCompensation() = default;

    ~GlobalMotionCompensation();

    Eigen::MatrixXf apply(const cv::Mat &frame_raw);
};