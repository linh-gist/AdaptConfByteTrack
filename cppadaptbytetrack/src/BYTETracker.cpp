#include "BYTETracker.h"
#include <fstream>

/**
 * @brief Constructs a BYTETracker object with specified tracking parameters.
 *
 * This constructor initializes a BYTETracker instance for multi-object tracking, setting thresholds for track detection and matching,
 * configuring the track buffer based on frame rate, and enabling or disabling global motion compensation (GMC).
 *
 * @param frate The frame rate of the video (frames per second).
 * @param tbuffer The track buffer duration in seconds, defining how long a track can remain unmatched before being removed.
 * @param tthresh The threshold for track detection confidence. It is a threshold to specified low/high confidence tracks
 * @param mthresh The threshold for matching tracks to detections; matches with costs above this are rejected.
 * @param use_gmc Boolean flag indicating whether to enable global motion compensation to account for camera motion.
 */
BYTETracker::BYTETracker(int frate, int tbuffer, float tthresh, float mthresh, bool use_gmc) {
    this->track_thresh = tthresh;
    this->high_thresh = tthresh + 0.1;
    this->match_thresh = mthresh;

    frame_id = 0;
    max_time_lost = int(frate / 30.0 * tbuffer); // track buffer
    // cout << "Init ByteTrack!" << endl;
    _gmc_enabled = use_gmc;
    _gmc_algo = GlobalMotionCompensation();
}

BYTETracker::~BYTETracker() {
}

/**
 * @brief Updates the tracker's state with new detections and returns active tracks.
 *
 * This function processes a new frame's detections, applying global motion compensation (if enabled), associating detections with
 * existing tracks using IoU-based matching, updating track states, and managing track lifecycles (tracked, lost, removed). It performs
 * multiple association steps to handle high-confidence and low-confidence detections, initializes new tracks, and removes outdated ones.
 * The function is central to this algorithm for multi-object tracking.
 *
 * @param objects A 2D vector of detections, where each detection is [left, top, right, bottom, confidence_score].
 * @param img_path Path to the current frame's image file, used for global motion compensation if enabled.
 *
 * @return A 2D vector of active tracks, where each track is [track_id, top, left, width, height].
 */
vector <vector<float>> BYTETracker::update(const vector <vector<float>> &objects, string img_path) {
    ////////////////// Camera Motion Compensation
    Eigen::MatrixXf M = Eigen::MatrixXf::Zero(8, 9);
    M.setIdentity();
    if (_gmc_enabled) {
        cv::Mat img = cv::imread(img_path);
        Eigen::MatrixXf H = _gmc_algo.apply(img);
        M(0, 0) = H(0, 0);
        M(0, 1) = H(0, 1);
        M(1, 0) = H(1, 0);
        M(1, 1) = H(1, 1);
        M(2, 2) = 1;
        M(6, 6) = 1;
        float height_trans = sqrt(pow(H(0, 1), 2) + pow(H(1, 1), 2));
        M(3, 3) = height_trans;
        M(7, 7) = height_trans;
        M(4, 4) = H(0, 0);
        M(4, 5) = H(0, 1);
        M(5, 4) = H(1, 0);
        M(5, 5) = H(1, 1);
        M(0, 8) = H(0, 2);
        M(1, 8) = H(1, 2);
        //cout<<H<<endl<<endl<<M<<endl<<" > endl";
    }

    // objects[i] : Left, Top, Right, Bottom, Conf
    ////////////////// Transform Input - Adaptive confidence ////
    float threshold = track_thresh;
    if (objects.size() > 1) {
        // Compute differences between consecutive elements
        std::vector<float> differences(objects.size() - 1);
        for (size_t i = 0; i < objects.size() - 1; ++i) {
            differences[i] = objects[i + 1][4] - objects[i][4];
        }
        // Find the index of the minimum difference
        auto min_diff_iter = std::min_element(differences.begin(), differences.end());
        size_t min_diff_index = std::distance(differences.begin(), min_diff_iter);
        // Get the threshold value
        threshold = objects[min_diff_index][4];
        if (threshold < high_thresh) {
            threshold = high_thresh;
        }
    }
    ////////////////// Step 1: Get detections //////////////////
    this->frame_id++;
    vector <STrack> activated_stracks;
    vector <STrack> refind_stracks;
    vector <STrack> removed_stracks;
    vector <STrack> lost_stracks;
    vector <STrack> detections;
    vector <STrack> detections_low;

    vector <STrack> detections_cp;
    vector <STrack> tracked_stracks_swap;
    vector <STrack> resa, resb;
    vector <vector<float>> output_stracks;

    vector < STrack * > unconfirmed;
    vector < STrack * > tracked_stracks;
    vector < STrack * > strack_pool;
    vector < STrack * > r_tracked_stracks;

    if (objects.size() > 0) {
        for (int i = 0; i < objects.size(); i++) {
            vector<float> tlbr_;
            tlbr_.resize(4);
            tlbr_[0] = objects[i][0];
            tlbr_[1] = objects[i][1];
            tlbr_[2] = objects[i][2];
            tlbr_[3] = objects[i][3];

            float score = objects[i][4];

            STrack strack(STrack::tlbr_to_tlwh(tlbr_), score);
            if (score >= threshold) // track_thresh
            {
                detections.push_back(strack);
            } else if (score > 0.1) {
                detections_low.push_back(strack);
            }

        }
    }

    // Add newly detected tracklets to tracked_stracks
    for (int i = 0; i < this->tracked_stracks.size(); i++) {
        if (!this->tracked_stracks[i].is_activated)
            unconfirmed.push_back(&this->tracked_stracks[i]);
        else
            tracked_stracks.push_back(&this->tracked_stracks[i]);
    }

    ////////////////// Step 2: First association, with IoU //////////////////
    strack_pool = joint_stracks(tracked_stracks, this->lost_stracks);
    STrack::multi_predict(strack_pool, this->kalman_filter, M);

    vector <vector<float>> dists;
    int dist_size = 0, dist_size_size = 0;
    dists = iou_distance(strack_pool, detections, dist_size, dist_size_size);

    vector <vector<int>> matches;
    vector<int> u_track, u_detection;
    linear_assignment(dists, dist_size, dist_size_size, match_thresh, matches, u_track, u_detection);

    for (int i = 0; i < matches.size(); i++) {
        STrack *track = strack_pool[matches[i][0]];
        STrack *det = &detections[matches[i][1]];
        if (track->state == TrackState::Tracked) {
            track->update(*det, this->frame_id);
            activated_stracks.push_back(*track);
        } else {
            track->re_activate(*det, this->frame_id, false);
            refind_stracks.push_back(*track);
        }
    }

    ////////////////// Step 3: Second association, using low score dets //////////////////
    for (int i = 0; i < u_detection.size(); i++) {
        detections_cp.push_back(detections[u_detection[i]]);
    }
    detections.clear();
    detections.assign(detections_low.begin(), detections_low.end());

    for (int i = 0; i < u_track.size(); i++) {
        if (strack_pool[u_track[i]]->state == TrackState::Tracked) {
            r_tracked_stracks.push_back(strack_pool[u_track[i]]);
        }
    }

    dists.clear();
    dists = iou_distance(r_tracked_stracks, detections, dist_size, dist_size_size);

    matches.clear();
    u_track.clear();
    u_detection.clear();
    linear_assignment(dists, dist_size, dist_size_size, 0.5, matches, u_track, u_detection);

    for (int i = 0; i < matches.size(); i++) {
        STrack *track = r_tracked_stracks[matches[i][0]];
        STrack *det = &detections[matches[i][1]];
        if (track->state == TrackState::Tracked) {
            track->update(*det, this->frame_id);
            activated_stracks.push_back(*track);
        } else {
            track->re_activate(*det, this->frame_id, false);
            refind_stracks.push_back(*track);
        }
    }

    for (int i = 0; i < u_track.size(); i++) {
        STrack *track = r_tracked_stracks[u_track[i]];
        if (track->state != TrackState::Lost) {
            track->mark_lost();
            lost_stracks.push_back(*track);
        }
    }

    // Deal with unconfirmed tracks, usually tracks with only one beginning frame
    detections.clear();
    detections.assign(detections_cp.begin(), detections_cp.end());

    dists.clear();
    dists = iou_distance(unconfirmed, detections, dist_size, dist_size_size);

    matches.clear();
    vector<int> u_unconfirmed;
    u_detection.clear();
    linear_assignment(dists, dist_size, dist_size_size, 0.7, matches, u_unconfirmed, u_detection);

    for (int i = 0; i < matches.size(); i++) {
        unconfirmed[matches[i][0]]->update(detections[matches[i][1]], this->frame_id);
        activated_stracks.push_back(*unconfirmed[matches[i][0]]);
    }

    for (int i = 0; i < u_unconfirmed.size(); i++) {
        STrack *track = unconfirmed[u_unconfirmed[i]];
        track->mark_removed();
        removed_stracks.push_back(*track);
    }

    ////////////////// Step 4: Init new stracks //////////////////
    for (int i = 0; i < u_detection.size(); i++) {
        STrack *track = &detections[u_detection[i]];
        if (track->score < this->high_thresh)
            continue;
        track->activate(this->kalman_filter, this->frame_id);
        activated_stracks.push_back(*track);
    }

    ////////////////// Step 5: Update state //////////////////
    for (int i = 0; i < this->lost_stracks.size(); i++) {
        if (this->frame_id - this->lost_stracks[i].end_frame() > this->max_time_lost) {
            this->lost_stracks[i].mark_removed();
            removed_stracks.push_back(this->lost_stracks[i]);
        }
    }

    for (int i = 0; i < this->tracked_stracks.size(); i++) {
        if (this->tracked_stracks[i].state == TrackState::Tracked) {
            tracked_stracks_swap.push_back(this->tracked_stracks[i]);
        }
    }
    this->tracked_stracks.clear();
    this->tracked_stracks.assign(tracked_stracks_swap.begin(), tracked_stracks_swap.end());

    this->tracked_stracks = joint_stracks(this->tracked_stracks, activated_stracks);
    this->tracked_stracks = joint_stracks(this->tracked_stracks, refind_stracks);

    //std::cout << activated_stracks.size() << std::endl;

    this->lost_stracks = sub_stracks(this->lost_stracks, this->tracked_stracks);
    for (int i = 0; i < lost_stracks.size(); i++) {
        this->lost_stracks.push_back(lost_stracks[i]);
    }

    this->lost_stracks = sub_stracks(this->lost_stracks, this->removed_stracks);
    for (int i = 0; i < removed_stracks.size(); i++) {
        this->removed_stracks.push_back(removed_stracks[i]);
    }

    remove_duplicate_stracks(resa, resb, this->tracked_stracks, this->lost_stracks);

    this->tracked_stracks.clear();
    this->tracked_stracks.assign(resa.begin(), resa.end());
    this->lost_stracks.clear();
    this->lost_stracks.assign(resb.begin(), resb.end());

    for (int i = 0; i < this->tracked_stracks.size(); i++) {
        if (this->tracked_stracks[i].is_activated) {
            STrack tmp = this->tracked_stracks[i];
            vector<float> id_ltrb = {(float) tmp.track_id, tmp.tlwh[0], tmp.tlwh[1], tmp.tlwh[2], tmp.tlwh[3]};
            output_stracks.push_back(id_ltrb);
        }
    }
    return output_stracks;
}