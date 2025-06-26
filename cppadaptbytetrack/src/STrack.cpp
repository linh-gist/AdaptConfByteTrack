#include "STrack.h"

/**
 * @brief Constructs an STrack object with initial bounding box and detection score.
 *
 * This constructor initializes a new track with a bounding box in top-left-width-height (tlwh) format and a detection confidence score.
 * Initializing properties like track ID, state, and frame information. The track starts as unactivated and in the "New" state, ready for further processing.
 *
 * @param tlwh_ A vector of four floats representing the bounding box in [top, left, width, height] format.
 * @param score The confidence score of the detection associated with this track, typically from a detector.
 */
STrack::STrack(vector<float> tlwh_, float score) {
    _tlwh.resize(4);
    _tlwh.assign(tlwh_.begin(), tlwh_.end());

    is_activated = false;
    track_id = 0;
    state = TrackState::New;

    tlwh.resize(4);
    tlbr.resize(4);

    static_tlwh();
    static_tlbr();
    frame_id = 0;
    tracklet_len = 0;
    this->score = score;
    start_frame = 0;
}

STrack::~STrack() {
}

/**
 * @brief Activates a track by initializing its Kalman filter and setting its vector state.
 *
 * This function activates a new track by assigning it a unique track ID, initializing the Kalman filter with the track's bounding box
 * in center-based coordinates (xyah where x&y are center of a bbox, a=w/h) and updating its state to "Tracked".
 * It is typically called when a new detection is confirmed as a track. The track's initial frame and other properties are also set.
 *
 * @param kalman_filter Reference to a KalmanFilter object used to initialize the track's state estimation.
 * @param frame_id The ID of the current frame in the video sequence, used to set the track's starting frame.
 */
void STrack::activate(byte_kalman::KalmanFilter &kalman_filter, int frame_id) {
    this->kalman_filter = kalman_filter;
    this->track_id = this->next_id();

    vector<float> _tlwh_tmp(4);
    _tlwh_tmp[0] = this->_tlwh[0];
    _tlwh_tmp[1] = this->_tlwh[1];
    _tlwh_tmp[2] = this->_tlwh[2];
    _tlwh_tmp[3] = this->_tlwh[3];
    vector<float> xyah = tlwh_to_xyah(_tlwh_tmp);
    DETECTBOX xyah_box;
    xyah_box[0] = xyah[0];
    xyah_box[1] = xyah[1];
    xyah_box[2] = xyah[2];
    xyah_box[3] = xyah[3];
    auto mc = this->kalman_filter.initiate(xyah_box);
    this->mean = mc.first;
    this->covariance = mc.second;

    static_tlwh();
    static_tlbr();

    this->tracklet_len = 0;
    this->state = TrackState::Tracked;
    if (frame_id == 1) {
        this->is_activated = true;
    }
    //this->is_activated = true;
    this->frame_id = frame_id;
    this->start_frame = frame_id;
}

/**
 * @brief Re-activates an existing track with updated information from a new detection.
 *
 * This function updates an existing track's state using a new detection's bounding box and score. It updates the Kalman filter
 * with the new bounding box in center-based coordinates, resets the tracklet length, and sets the track's state to "Tracked".
 * Optionally, it assigns a new track ID if requested. This is used to revive a track that was temporarily lost or unmatched.
 *
 * @param new_track Reference to an STrack object containing the new detection's bounding box and score.
 * @param frame_id The ID of the current frame in the video sequence, used to update the track's frame information.
 * @param new_id Boolean indicating whether to assign a new track ID (true) or retain the existing one (false).
 */
void STrack::re_activate(STrack &new_track, int frame_id, bool new_id) {
    vector<float> xyah = tlwh_to_xyah(new_track.tlwh);
    DETECTBOX xyah_box;
    xyah_box[0] = xyah[0];
    xyah_box[1] = xyah[1];
    xyah_box[2] = xyah[2];
    xyah_box[3] = xyah[3];
    auto mc = this->kalman_filter.update(this->mean, this->covariance, xyah_box);
    this->mean = mc.first;
    this->covariance = mc.second;

    static_tlwh();
    static_tlbr();

    this->tracklet_len = 0;
    this->state = TrackState::Tracked;
    this->is_activated = true;
    this->frame_id = frame_id;
    this->score = new_track.score;
    if (new_id)
        this->track_id = next_id();
}

/**
 * @brief Updates an existing track with new detection information.
 *
 * This function updates the track's state using a new detection's bounding box and score. It increments the tracklet length,
 * updates the Kalman filter with the new bounding box in center-based coordinates (xyah), and sets the track's state to "Tracked"
 * and activated. This is used to refine a track's position and attributes when matched with a new detection.
 *
 * @param new_track Reference to an STrack object containing the new detection's bounding box and score.
 * @param frame_id The ID of the current frame in the video sequence, used to update the track's frame information.
 */
void STrack::update(STrack &new_track, int frame_id) {
    this->frame_id = frame_id;
    this->tracklet_len++;

    vector<float> xyah = tlwh_to_xyah(new_track.tlwh);
    DETECTBOX xyah_box;
    xyah_box[0] = xyah[0];
    xyah_box[1] = xyah[1];
    xyah_box[2] = xyah[2];
    xyah_box[3] = xyah[3];

    auto mc = this->kalman_filter.update(this->mean, this->covariance, xyah_box);
    this->mean = mc.first;
    this->covariance = mc.second;

    static_tlwh();
    static_tlbr();

    this->state = TrackState::Tracked;
    this->is_activated = true;

    this->score = new_track.score;
}

/**
 * @brief Updates the track's bounding box in top-left-width-height (tlwh) format based on its state.
 *
 * This function sets the track's tlwh member (top-left-width-height bounding box) based on its current state. For new tracks,
 * it copies the initial _tlwh values directly. It converts the Kalman filter's mean state (assumed to be in center-based xyah format:
 * center-x, center-y, aspect ratio (width / height), height) to tlwh format. This is used to maintain consistent bounding box representation.
 */
void STrack::static_tlwh() {
    if (this->state == TrackState::New) {
        tlwh[0] = _tlwh[0];
        tlwh[1] = _tlwh[1];
        tlwh[2] = _tlwh[2];
        tlwh[3] = _tlwh[3];
        return;
    }

    tlwh[0] = mean[0];
    tlwh[1] = mean[1];
    tlwh[2] = mean[2];
    tlwh[3] = mean[3];

    tlwh[2] *= tlwh[3];
    tlwh[0] -= tlwh[2] / 2;
    tlwh[1] -= tlwh[3] / 2;
}

/**
 * This function converts the track's bounding box from top-left-width-height (tlwh) format to top-left-bottom-right (tlbr)
 */
void STrack::static_tlbr() {
    tlbr.clear();
    tlbr.assign(tlwh.begin(), tlwh.end());
    tlbr[2] += tlbr[0];
    tlbr[3] += tlbr[1];
}

/**
 * @brief Converts a bounding box from top-left-width-height (tlwh) to center-x, center-y, aspect-ratio (w/h), height (xyah) format.
 *
 * This function transforms a bounding box from tlwh format [top, left, width, height] to xyah format [center_x, center_y, aspect_ratio, height].
 *
 * @param tlwh_tmp A vector of four floats representing the bounding box in [top, left, width, height] format.
 * @return A vector of four floats representing the bounding box in [center_x, center_y, aspect_ratio, height] format.
 */
vector<float> STrack::tlwh_to_xyah(vector<float> tlwh_tmp) {
    vector<float> tlwh_output = tlwh_tmp;
    tlwh_output[0] += tlwh_output[2] / 2;
    tlwh_output[1] += tlwh_output[3] / 2;
    tlwh_output[2] /= tlwh_output[3];
    return tlwh_output;
}

vector<float> STrack::to_xyah() {
    return tlwh_to_xyah(tlwh);
}

// convert vector state format from top-left-bottom-right to top-left-width-height
vector<float> STrack::tlbr_to_tlwh(vector<float> &tlbr) {
    tlbr[2] -= tlbr[0];
    tlbr[3] -= tlbr[1];
    return tlbr;
}

// mark a TrackState to 'Lost'
void STrack::mark_lost() {
    state = TrackState::Lost;
}

// mark a TrackState to 'Removed'
void STrack::mark_removed() {
    state = TrackState::Removed;
}

// generate the next unique track ID for a new track.
int STrack::next_id() {
    static int _count = 0;
    _count++;
    return _count;
}

// returns the most recent frame ID associated with the track.
int STrack::end_frame() {
    return this->frame_id;
}

/**
 * @brief Performs motion prediction for multiple tracks using a Kalman filter and global motion compensation.
 *
 * This function updates the state (mean and covariance) of multiple tracks by applying a global motion transformation (camera motion compensation)
 * and then performing a Kalman filter prediction step. It is used to predict the next position
 * of tracks across frames, accounting for global motion. The bounding box representations (tlwh and tlbr) are updated accordingly.
 *
 * @param stracks Vector of pointers to STrack objects representing the tracks to be predicted.
 * @param kalman_filter Reference to a KalmanFilter object used for state prediction.
 * @param M Eigen matrix representing the global motion model, calculated in BYTETracker::update(...)
 */
void STrack::multi_predict(vector<STrack *> &stracks, byte_kalman::KalmanFilter &kalman_filter, Eigen::MatrixXf M) {
    Eigen::MatrixXf m_homo = M(Eigen::all, Eigen::seq(0, 7));
    Eigen::VectorXf m_trans = M.col(8);
    for (int i = 0; i < stracks.size(); i++) {
        if (stracks[i]->state != TrackState::Tracked) {
            stracks[i]->mean[7] = 0;
        }
        stracks[i]->mean = stracks[i]->mean * m_homo.transpose() + m_trans.transpose();
        stracks[i]->covariance = m_homo * stracks[i]->covariance * m_homo.transpose();
        kalman_filter.predict(stracks[i]->mean, stracks[i]->covariance);
        stracks[i]->static_tlwh();
        stracks[i]->static_tlbr();
    }
}