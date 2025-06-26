#pragma once

#include "dataType.h"

namespace byte_kalman {
    class KalmanFilter {
    public:
        /*
        * This array provides critical values for the chi-squared distribution at a 95% confidence level for
        * degrees of freedom from 1 to 10. These values are used as thresholds for gating in the tracking algorithm
        * to validate measurements. For example, in a Kalman filter, the Mahalanobis distance between a predicted
        * state and a measurement is compared against these thresholds to determine if the measurement lies within
        * the 95% confidence ellipse.
         */
        static const double chi2inv95[10];

        KalmanFilter();

        KAL_DATA initiate(const DETECTBOX &measurement);

        void predict(KAL_MEAN &mean, KAL_COVA &covariance);

        KAL_HDATA project(const KAL_MEAN &mean, const KAL_COVA &covariance);

        KAL_DATA update(const KAL_MEAN &mean,
                        const KAL_COVA &covariance,
                        const DETECTBOX &measurement);

        Eigen::Matrix<float, 1, -1> gating_distance(
                const KAL_MEAN &mean,
                const KAL_COVA &covariance,
                const std::vector <DETECTBOX> &measurements,
                bool only_position = false);

    private:
        // Defines the state transition model in the Kalman filter, mapping the current state (e.g., position, velocity)
        // to the next time step. Stored as an 8x8 row-major matrix
        Eigen::Matrix<float, 8, 8, Eigen::RowMajor> _motion_mat;
        // Maps the state vector (8D) to the measurement space (4D, e.g., position measurements) in the Kalman filter.
        // This is called Projection Matrix, Used to compute predicted measurements.
        Eigen::Matrix<float, 4, 8, Eigen::RowMajor> _update_mat;
        // Scales the process noise covariance for position components in the Kalman filter. Determines how much
        // uncertainty is assumed in the position prediction due to process noise.
        float _std_weight_position;
        // Scales the process noise covariance for velocity components in the Kalman filter. Controls the assumed
        // uncertainty in velocity predictions due to process noise.
        float _std_weight_velocity;
    };
}