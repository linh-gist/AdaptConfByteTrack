#include "kalmanFilter.h"
#include <Eigen/Cholesky>

namespace byte_kalman {
    /* @note The values are typically derived from statistical tables or computed using a chi-squared inverse CDF
     * function (e.g., from libraries like Boost or scipy.stats in Python for reference).
     * If gating dimension is 2 we should use chi2inv95[2], If gating dimension is 4 (bbox), we should use chi2inv95[4]
     */
    const double KalmanFilter::chi2inv95[10] = {
            0,// 0 degree of freedom: 95% confidence threshold
            3.8415,// 1 degrees of freedom
            5.9915,// 2 degrees of freedom
            7.8147,// 3 degrees of freedom
            9.4877,// 4 degrees of freedom
            11.070,// 5 degrees of freedom
            12.592,// 6 degrees of freedom
            14.067,// 7 degrees of freedom
            15.507,// 8 degrees of freedom
            16.919// 9 degrees of freedom
    };

    KalmanFilter::KalmanFilter() {
        int ndim = 4; // dimension of detection (measurement), top-left-aspect_ratio-height
        double dt = 1.; // sampling period, frame interval assuming consistent frame-by-frame updates.

        _motion_mat = Eigen::MatrixXf::Identity(8, 8); // motion matrix
        for (int i = 0; i < ndim; i++) {
            _motion_mat(i, ndim + i) = dt;
        }
        _update_mat = Eigen::MatrixXf::Identity(4, 8); // projection matrix, from vector state to measurement/detection
        // Weight for position uncertainty in Kalman filter, set to 1/20 for balancing moderate position noise scaling.
        // Constant: 1./20 (Scales position noise in process covariance;)
        this->_std_weight_position = 1. / 20;
        // Weight for velocity uncertainty in Kalman filter, set to 1/160 for lower velocity noise scaling.
        this->_std_weight_velocity = 1. / 160;
    }

    /**
     * @brief Initializes the Kalman filter with an initial measurement for a new track.
     *
     * This function sets up the initial state (mean and covariance) of the Kalman filter for a new track based on a detection's
     * bounding box in center-based format (e.g., [center_x, center_y, aspect_ratio (w/h), height]). It initializes the position components
     * from the measurement and sets velocity components to zero, with predefined uncertainties for position and velocity.
     *
     * @param measurement A DETECTBOX (4D vector) containing the initial bounding box in [center_x, center_y, aspect_ratio, height] format.
     * @return A pair containing the initial state mean (KAL_MEAN, 8D vector: 4 position + 4 velocity components) and covariance matrix
     *         (KAL_COVA, 8x8 diagonal matrix) for the Kalman filter.
     */
    KAL_DATA KalmanFilter::initiate(const DETECTBOX &measurement) {
        DETECTBOX mean_pos = measurement;
        DETECTBOX mean_vel;
        for (int i = 0; i < 4; i++) mean_vel(i) = 0;

        KAL_MEAN mean;
        for (int i = 0; i < 8; i++) {
            if (i < 4) mean(i) = mean_pos(i);
            else mean(i) = mean_vel(i - 4);
        }

        KAL_MEAN std;
        std(0) = 2 * _std_weight_position * measurement[3];
        std(1) = 2 * _std_weight_position * measurement[3];
        std(2) = 1e-2;
        std(3) = 2 * _std_weight_position * measurement[3];
        std(4) = 10 * _std_weight_velocity * measurement[3];
        std(5) = 10 * _std_weight_velocity * measurement[3];
        std(6) = 1e-5;
        std(7) = 10 * _std_weight_velocity * measurement[3];

        KAL_MEAN tmp = std.array().square();
        KAL_COVA var = tmp.asDiagonal();
        return std::make_pair(mean, var);
    }

    /**
     * @brief Performs the prediction step of the Kalman filter.
     *
     * This function predicts the next state and covariance of a track using the Kalman filter's motion model (constant veloity). It applies the motion
     * transition matrix to the current state mean and covariance, adding process noise to account for uncertainties in position and
     * velocity. The prediction is used to estimate the track's state in the next frame before incorporating new measurements.
     *
     * @param mean Input and output parameter: The current state mean (8D vector: 4 position + 4 velocity components). Updated to
     *             the predicted state mean after the function executes.
     * @param covariance Input and output parameter: The current state covariance (8x8 matrix). Updated to the predicted state
     *                   covariance after the function executes.
     */
    void KalmanFilter::predict(KAL_MEAN &mean, KAL_COVA &covariance) {
        //revise the data;
        DETECTBOX std_pos;
        std_pos << _std_weight_position * mean(3),
                _std_weight_position * mean(3),
                1e-2,
                _std_weight_position * mean(3);
        DETECTBOX std_vel;
        std_vel << _std_weight_velocity * mean(3),
                _std_weight_velocity * mean(3),
                1e-5,
                _std_weight_velocity * mean(3);
        KAL_MEAN tmp;
        tmp.block<1, 4>(0, 0) = std_pos;
        tmp.block<1, 4>(0, 4) = std_vel;
        tmp = tmp.array().square();
        KAL_COVA motion_cov = tmp.asDiagonal();
        KAL_MEAN mean1 = this->_motion_mat * mean.transpose();
        KAL_COVA covariance1 = this->_motion_mat * covariance * (_motion_mat.transpose());
        covariance1 += motion_cov;

        mean = mean1;
        covariance = covariance1;
    }

    /**
     * @brief Projects the Kalman filter state into the measurement space.
     *
     * This function maps the current state (mean and covariance) from the Kalman filter's state space (position and velocity)
     * to the measurement space (bounding box in [center_x, center_y, aspect_ratio (w/h), height]) using the update/projection matrix. It also
     * adds measurement noise to the projected covariance to account for detection uncertainties. This is used to prepare the update
     * step for comparison with new measurements.
     *
     * @param mean The current state mean (8D vector: 4 position + 4 velocity components).
     * @param covariance The current state covariance (8x8 matrix).
     *
     * @return A pair containing the projected mean (KAL_HMEAN, 4D vector in measurement space) and the projected covariance
     *         (KAL_HCOVA, 4x4 matrix) in the measurement space.
     */
    KAL_HDATA KalmanFilter::project(const KAL_MEAN &mean, const KAL_COVA &covariance) {
        DETECTBOX std;
        std << _std_weight_position * mean(3), _std_weight_position * mean(3),
                1e-1, _std_weight_position * mean(3);
        KAL_HMEAN mean1 = _update_mat * mean.transpose();
        KAL_HCOVA covariance1 = _update_mat * covariance * (_update_mat.transpose());
        Eigen::Matrix<float, 4, 4> diag = std.asDiagonal();
        diag = diag.array().square().matrix();
        covariance1 += diag;
        //    covariance1.diagonal() << diag;
        return std::make_pair(mean1, covariance1);
    }

    /**
     * @brief Performs the update step of the Kalman filter with a new measurement/detection.
     *
     * This function updates the Kalman filter's state (mean and covariance) by incorporating a new measurement (a detected bounding box).
     * It projects the current state into the measurement space, computes the Kalman gain, and corrects the state based on the difference
     * between the measurement and the projected state (so called innovation). This is used to refine
     * a track's state estimate with new detection data.
     *
     * @param mean The current state mean (8D vector: 4 position + 4 velocity components).
     * @param covariance The current state covariance (8x8 matrix).
     * @param measurement A DETECTBOX (4D vector) containing the new measurement in [center_x, center_y, aspect_ratio (w/h), height] format.
     *
     * @return A pair containing the updated state mean (KAL_MEAN, 8D vector) and covariance (KAL_COVA, 8x8 matrix).
     */
    KAL_DATA
    KalmanFilter::update(
            const KAL_MEAN &mean,
            const KAL_COVA &covariance,
            const DETECTBOX &measurement) {
        KAL_HDATA pa = project(mean, covariance);
        KAL_HMEAN projected_mean = pa.first;
        KAL_HCOVA projected_cov = pa.second;

        //chol_factor, lower =
        //scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        //kalmain_gain =
        //scipy.linalg.cho_solve((cho_factor, lower),
        //np.dot(covariance, self._upadte_mat.T).T,
        //check_finite=False).T
        Eigen::Matrix<float, 4, 8> B = (covariance * (_update_mat.transpose())).transpose();
        Eigen::Matrix<float, 8, 4> kalman_gain = (projected_cov.llt().solve(B)).transpose(); // eg.8x4
        Eigen::Matrix<float, 1, 4> innovation = measurement - projected_mean; //eg.1x4
        auto tmp = innovation * (kalman_gain.transpose());
        KAL_MEAN new_mean = (mean.array() + tmp.array()).matrix();
        KAL_COVA new_covariance = covariance - kalman_gain * projected_cov * (kalman_gain.transpose());
        return std::make_pair(new_mean, new_covariance);
    }

    /**
     * @brief Computes the squared Mahalanobis distance between the predicted state and multiple measurements.
     *
     * This function calculates the squared Mahalanobis distance between the Kalman filter's projected state (mean and covariance)
     * and a set of measurements (bounding boxes). The Mahalanobis distance is used to gate measurements,
     * identifying which detections are likely associated with a track based on their statistical distance.
     * Currently, it only supports full state measurements and exits if only position components are requested.
     *
     * @param mean The current state mean (8D vector: 4 position + 4 velocity components).
     * @param covariance The current state covariance (8x8 matrix).
     * @param measurements A vector of DETECTBOX objects, each a 4D vector representing a bounding box
     *                     in [center_x, center_y, aspect_ratio, height] format.
     * @param only_position Boolean flag indicating whether to consider only position components (true) or the full state (false).
     *
     * @return A row vector (Eigen::Matrix<float, 1, -1>) containing the squared Mahalanobis distances for each measurement.
     */
    Eigen::Matrix<float, 1, -1>
    KalmanFilter::gating_distance(
            const KAL_MEAN &mean,
            const KAL_COVA &covariance,
            const std::vector <DETECTBOX> &measurements,
            bool only_position) {
        KAL_HDATA pa = this->project(mean, covariance);
        if (only_position) {
            printf("not implement!");
            exit(0);
        }
        KAL_HMEAN mean1 = pa.first;
        KAL_HCOVA covariance1 = pa.second;

        //    Eigen::Matrix<float, -1, 4, Eigen::RowMajor> d(size, 4);
        DETECTBOXSS d(measurements.size(), 4);
        int pos = 0;
        for (DETECTBOX box : measurements) {
            d.row(pos++) = box - mean1;
        }
        Eigen::Matrix<float, -1, -1, Eigen::RowMajor> factor = covariance1.llt().matrixL();
        Eigen::Matrix<float, -1, -1> z = factor.triangularView<Eigen::Lower>().solve<Eigen::OnTheRight>(d).transpose();
        auto zz = ((z.array()) * (z.array())).matrix();
        auto square_maha = zz.colwise().sum();
        return square_maha;
    }
}