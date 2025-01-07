#include "GlobalMotionCompensation.h"

GlobalMotionCompensation::~GlobalMotionCompensation() {}

// Affine Motion Estimation or Affine Global Motion Estimation using Sparse Optical Flow
Eigen::MatrixXf GlobalMotionCompensation::apply(const cv::Mat &frame_raw) {
    // Initialization
    int height = frame_raw.rows;
    int width = frame_raw.cols;
    Eigen::MatrixXf H = Eigen::MatrixXf::Zero(2, 3);
    H.setIdentity();
    cv::Mat frame;
    cv::cvtColor(frame_raw, frame, cv::COLOR_BGR2GRAY);

    // Downscale
    if (_downscale > 1.0F) {
        width /= _downscale, height /= _downscale;
        cv::resize(frame, frame, cv::Size(width, height));
    }

    if (!_first_frame_initialized) {
        /**
         *  If this is the first frame, there is nothing to match
         *  Save the keypoints and descriptors, return identity matrix
         */
        _first_frame_initialized = true;
        _prev_frame = frame.clone();
        return H;
    }
    // Detect keypoints
    std::vector<cv::Point2f> prev_pts;
    cv::goodFeaturesToTrack(_prev_frame, prev_pts, 200, 0.01, 30, cv::noArray(), 3);

    // Find correspondences between the previous and current frame
    std::vector<cv::Point2f> matched_keypoints;
    std::vector<uchar> status;
    std::vector<float> err;
    try {
        cv::calcOpticalFlowPyrLK(_prev_frame, frame, prev_pts, matched_keypoints, status, err);
    }
    catch (const cv::Exception &e) {
        std::cout << "Warning: Could not find correspondences for GMC" << std::endl;
        _prev_frame = frame.clone();
        return H;
    }
    // Keep good matches
    std::vector<cv::Point2f> prev_points, curr_points;
    for (size_t i = 0; i < matched_keypoints.size(); i++) {
        if (status[i]) {
            prev_points.push_back(prev_pts[i]);
            curr_points.push_back(matched_keypoints[i]);
        }
    }
    // Estimate affine matrix
    if (prev_points.size() > 4) {
        cv::Mat homography = cv::estimateAffine2D(prev_points, curr_points);
        //cv::Mat inliers;
        //cv::Mat homography = cv::findHomography(prev_points, curr_points, cv::RANSAC, 3,inliers, 500, 0.99);
        H << homography.at<double>(0, 0), homography.at<double>(0, 1), homography.at<double>(0, 2),
                homography.at<double>(1, 0), homography.at<double>(1, 1), homography.at<double>(1, 2);
    }
    _prev_frame = frame.clone();
    return H;
}