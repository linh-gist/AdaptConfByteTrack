#pragma once

#include <opencv2/opencv.hpp>
#include "kalmanFilter.h"

using namespace cv;
using namespace std;

// enum TrackState: Defines the possible states of a tracking object in the system.
// - New: Initial state when a track is first detected but not yet confirmed.
// - Tracked: State when the track is actively being followed and updated.
// - Lost: State when the track is no longer detected but not yet removed.
// - Removed: Final state when the track is terminated and removed from the system.
enum TrackState {
    New = 0, Tracked, Lost, Removed
};

class STrack {
public:
    STrack(vector<float> tlwh_, float score);

    ~STrack();

    vector<float> static tlbr_to_tlwh(vector<float> &tlbr);

    void static multi_predict(vector<STrack *> &stracks, byte_kalman::KalmanFilter &kalman_filter, Eigen::MatrixXf M);

    void static_tlwh();

    void static_tlbr();

    vector<float> tlwh_to_xyah(vector<float> tlwh_tmp);

    vector<float> to_xyah();

    void mark_lost();

    void mark_removed();

    int next_id();

    int end_frame();

    void activate(byte_kalman::KalmanFilter &kalman_filter, int frame_id);

    void re_activate(STrack &new_track, int frame_id, bool new_id = false);

    void update(STrack &new_track, int frame_id);

public:
    bool is_activated; // true if an object has a bbox association in the previous time step
    int track_id; // This integer assigns a distinct ID to each track
    int state; // one of TrackState{New = 0, Tracked, Lost, Removed}

    vector<float> _tlwh; // location and object size, Top, Left, Width, Height in image, copy of 'tlwh'
    vector<float> tlwh; // location and object size, Top, Left, Width, Height in image
    vector<float> tlbr; // location and object size, Top, Left, Bottom, Right in image
    int frame_id; // Frame index
    int tracklet_len; // survival length of an object
    int start_frame; // the frame index when the object borns

    KAL_MEAN mean; // vector state of an object
    KAL_COVA covariance; // covariance of an object
    float score; // confidence score of an object being tracked

private:
    byte_kalman::KalmanFilter kalman_filter; // each object is being traced by a Kalman Filter
};