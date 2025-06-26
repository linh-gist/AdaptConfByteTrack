#include "BYTETracker.h"
#include "lapjv.h"

/**
 * @brief Combines two sets of tracks into a single set, avoiding duplicate track IDs.
 *
 * This function combines two vectors of tracks, including all tracks from tlista (pointers to STrack objects) and adding tracks from tlistb
 * (STrack objects) only if their track IDs are not already present in tlista. It ensures a unified set of tracks for tracking algorithms
 * without duplicates, returning pointers to the tracks.
 *
 * @param tlista Vector of pointers to STrack objects representing the first set of tracks.
 * @param tlistb Vector of STrack objects representing the second set of tracks to be added.
 *
 * @return A vector of pointers to STrack objects containing all tracks from tlista and non-duplicate tracks from tlistb based on track IDs.
 */
vector<STrack *> BYTETracker::joint_stracks(vector<STrack *> &tlista, vector <STrack> &tlistb) {
    map<int, int> exists;
    vector < STrack * > res;
    for (int i = 0; i < tlista.size(); i++) {
        exists.insert(pair<int, int>(tlista[i]->track_id, 1));
        res.push_back(tlista[i]);
    }
    for (int i = 0; i < tlistb.size(); i++) {
        int tid = tlistb[i].track_id;
        if (!exists[tid] || exists.count(tid) == 0) {
            exists[tid] = 1;
            res.push_back(&tlistb[i]);
        }
    }
    return res;
}

/**
 * @brief Combines two sets of tracks into a single set, ensuring no duplicate track IDs.
 *
 * This function merges two vectors of STrack objects, including tracks from tlista and adding tracks from tlistb only if their track IDs
 * are not already present in tlista. It is used in tracking algorithms to create a unified set of tracks while avoiding duplicates.
 *
 * @param tlista Vector of STrack objects representing the first set of tracks (e.g., existing tracks).
 * @param tlistb Vector of STrack objects representing the second set of tracks to be added (e.g., new tracks).
 *
 * @return A vector of STrack objects containing all tracks from tlista and non-duplicate tracks from tlistb based on track IDs.
 */
vector <STrack> BYTETracker::joint_stracks(vector <STrack> &tlista, vector <STrack> &tlistb) {
    map<int, int> exists;
    vector <STrack> res;
    for (int i = 0; i < tlista.size(); i++) {
        exists.insert(pair<int, int>(tlista[i].track_id, 1));
        res.push_back(tlista[i]);
    }
    for (int i = 0; i < tlistb.size(); i++) {
        int tid = tlistb[i].track_id;
        if (!exists[tid] || exists.count(tid) == 0) {
            exists[tid] = 1;
            res.push_back(tlistb[i]);
        }
    }
    return res;
}

/**
 * @brief Computes the difference between two sets of tracks based on track IDs.
 *
 * This function returns a subset of tracks from tlista that are not present in tlistb, identified by their unique track IDs.
 * It filters out tracks from one set that overlap with another.
 *
 * @param tlista Vector of STrack objects representing the first set of tracks (all tracks).
 * @param tlistb Vector of STrack objects representing the second set of tracks (tracks to be subtracted).
 *
 * @return A vector of STrack objects containing tracks from tlista whose track IDs are not in tlistb.
 */
vector <STrack> BYTETracker::sub_stracks(vector <STrack> &tlista, vector <STrack> &tlistb) {
    map<int, STrack> stracks;
    for (int i = 0; i < tlista.size(); i++) {
        stracks.insert(pair<int, STrack>(tlista[i].track_id, tlista[i]));
    }
    for (int i = 0; i < tlistb.size(); i++) {
        int tid = tlistb[i].track_id;
        if (stracks.count(tid) != 0) {
            stracks.erase(tid);
        }
    }

    vector <STrack> res;
    std::map<int, STrack>::iterator it;
    for (it = stracks.begin(); it != stracks.end(); ++it) {
        res.push_back(it->second);
    }

    return res;
}

/**
 * @brief Removes duplicate tracks between two sets of tracks based on IoU distance and track age.
 *
 * This function identifies and removes duplicate tracks between two sets (stracksa and stracksb) by computing their IoU-based distances.
 * Tracks with an IoU distance below a threshold are considered potential duplicates, and the younger track (based on frame duration) is removed.
 * The function populates two output vectors with non-duplicate tracks.
 *
 * @param resa Output vector to store non-duplicate tracks from stracksa.
 * @param resb Output vector to store non-duplicate tracks from stracksb.
 * @param stracksa Input vector of STrack objects representing the first set of tracks.
 * @param stracksb Input vector of STrack objects representing the second set of tracks.
 */
void BYTETracker::remove_duplicate_stracks(vector <STrack> &resa, vector <STrack> &resb, vector <STrack> &stracksa,
                                           vector <STrack> &stracksb) {
    vector <vector<float>> pdist = iou_distance(stracksa, stracksb);
    vector <pair<int, int>> pairs;
    for (int i = 0; i < pdist.size(); i++) {
        for (int j = 0; j < pdist[i].size(); j++) {
            if (pdist[i][j] < 0.15) {
                pairs.push_back(pair<int, int>(i, j));
            }
        }
    }

    vector<int> dupa, dupb;
    for (int i = 0; i < pairs.size(); i++) {
        int timep = stracksa[pairs[i].first].frame_id - stracksa[pairs[i].first].start_frame;
        int timeq = stracksb[pairs[i].second].frame_id - stracksb[pairs[i].second].start_frame;
        if (timep > timeq)
            dupb.push_back(pairs[i].second);
        else
            dupa.push_back(pairs[i].first);
    }

    for (int i = 0; i < stracksa.size(); i++) {
        vector<int>::iterator iter = find(dupa.begin(), dupa.end(), i);
        if (iter == dupa.end()) {
            resa.push_back(stracksa[i]);
        }
    }

    for (int i = 0; i < stracksb.size(); i++) {
        vector<int>::iterator iter = find(dupb.begin(), dupb.end(), i);
        if (iter == dupb.end()) {
            resb.push_back(stracksb[i]);
        }
    }
}

/**
 * @brief Performs linear assignment on a cost matrix to match tracks and detections.
 *
 * This function uses the Jonker-Volgenant algorithm (via lapjv) to find optimal assignments between rows and columns of a cost matrix,
 * typically representing distances (e.g., IoU-based) between tracks and detections. It identifies matched pairs and unmatched elements
 * based on a cost threshold to associate detections with existing tracks.
 *
 * @param cost_matrix 2D vector representing the cost matrix, where cost_matrix[i][j] is the cost of assigning track i to detection j.
 * @param cost_matrix_size Number of rows in the cost matrix (number of tracks).
 * @param cost_matrix_size_size Number of columns in the cost matrix (number of detections).
 * @param thresh Maximum allowable cost for a valid assignment; assignments with costs above this are not considered.
 * @param matches Output vector of vectors, where each inner vector contains [row_index, col_index] for matched track-detection pairs.
 * @param unmatched_a Output vector containing indices of unmatched rows (unmatched tracks).
 * @param unmatched_b Output vector containing indices of unmatched columns (unmatched detections).
 */
void
BYTETracker::linear_assignment(vector <vector<float>> &cost_matrix, int cost_matrix_size, int cost_matrix_size_size,
                               float thresh,
                               vector <vector<int>> &matches, vector<int> &unmatched_a, vector<int> &unmatched_b) {
    if (cost_matrix.size() == 0) {
        for (int i = 0; i < cost_matrix_size; i++) {
            unmatched_a.push_back(i);
        }
        for (int i = 0; i < cost_matrix_size_size; i++) {
            unmatched_b.push_back(i);
        }
        return;
    }

    vector<int> rowsol;
    vector<int> colsol;
    float c = lapjv(cost_matrix, rowsol, colsol, true, thresh);
    for (int i = 0; i < rowsol.size(); i++) {
        if (rowsol[i] >= 0) {
            vector<int> match;
            match.push_back(i);
            match.push_back(rowsol[i]);
            matches.push_back(match);
        } else {
            unmatched_a.push_back(i);
        }
    }

    for (int i = 0; i < colsol.size(); i++) {
        if (colsol[i] < 0) {
            unmatched_b.push_back(i);
        }
    }
}

/**
 * @brief Computes the Intersection over Union (IoU) matrix for two sets of bounding boxes.
 *
 * This function calculates the IoU between pairs of bounding boxes from two sets, represented in top-left-bottom-right (tlbr) format.
 * IoU is a similarity metric used to measure the overlap between bounding boxes, to associate detections with tracks.
 *
 * @param atlbrs Vector of bounding boxes (e.g., existing tracks).
 * @param btlbrs Vector of bounding boxes (e.g., new detections).
 *
 * @return A 2D vector representing the IoU matrix, where element [i][j] is the IoU between atlbrs[i] and btlbrs[j].
 *         Returns an empty matrix if either input is empty.
 */
vector <vector<float>> BYTETracker::ious(vector <vector<float>> &atlbrs, vector <vector<float>> &btlbrs) {
    vector <vector<float>> ious;
    if (atlbrs.size() * btlbrs.size() == 0)
        return ious;

    ious.resize(atlbrs.size());
    for (int i = 0; i < ious.size(); i++) {
        ious[i].resize(btlbrs.size());
    }

    //bbox_ious
    for (int k = 0; k < btlbrs.size(); k++) {
        vector<float> ious_tmp;
        float box_area = (btlbrs[k][2] - btlbrs[k][0] + 1) * (btlbrs[k][3] - btlbrs[k][1] + 1);
        for (int n = 0; n < atlbrs.size(); n++) {
            float iw = min(atlbrs[n][2], btlbrs[k][2]) - max(atlbrs[n][0], btlbrs[k][0]) + 1;
            if (iw > 0) {
                float ih = min(atlbrs[n][3], btlbrs[k][3]) - max(atlbrs[n][1], btlbrs[k][1]) + 1;
                if (ih > 0) {
                    float ua =
                            (atlbrs[n][2] - atlbrs[n][0] + 1) * (atlbrs[n][3] - atlbrs[n][1] + 1) + box_area - iw * ih;
                    ious[n][k] = iw * ih / ua;
                } else {
                    ious[n][k] = 0.0;
                }
            } else {
                ious[n][k] = 0.0;
            }
        }
    }

    return ious;
}

/**
 * @brief Computes the IoU-based distance matrix between two sets of tracks.
 *
 * This function calculates the Intersection over Union (IoU) distance between two sets of tracks using their bounding boxes.
 * The IoU distance is defined as 1 - IoU, where lower IoU values indicate greater distance (less overlap).
 * The resulting cost matrix is used for track-to-detection assignment in tracking algorithms. It also updates the sizes of the input track sets.
 *
 * @param atracks Vector of pointers to STrack objects representing the first set of tracks (existing tracks).
 * @param btracks Vector of STrack objects representing the second set of tracks (new detections).
 * @param dist_size Output parameter to store the number of tracks in atracks.
 * @param dist_size_size Output parameter to store the number of tracks in btracks.
 *
 * @return A 2D vector representing the cost matrix, where each element [i][j] is the IoU distance (1 - IoU)
 *         between atracks[i] and btracks[j]. Returns empty matrix if either input is empty.
 */
vector <vector<float>>
BYTETracker::iou_distance(vector<STrack *> &atracks, vector <STrack> &btracks, int &dist_size, int &dist_size_size) {
    vector <vector<float>> cost_matrix;
    if (atracks.size() * btracks.size() == 0) {
        dist_size = atracks.size();
        dist_size_size = btracks.size();
        return cost_matrix;
    }
    vector <vector<float>> atlbrs, btlbrs;
    for (int i = 0; i < atracks.size(); i++) {
        atlbrs.push_back(atracks[i]->tlbr);
    }
    for (int i = 0; i < btracks.size(); i++) {
        btlbrs.push_back(btracks[i].tlbr);
    }

    dist_size = atracks.size();
    dist_size_size = btracks.size();

    vector <vector<float>> _ious = ious(atlbrs, btlbrs);

    for (int i = 0; i < _ious.size(); i++) {
        vector<float> _iou;
        for (int j = 0; j < _ious[i].size(); j++) {
            _iou.push_back(1 - _ious[i][j]);
        }
        cost_matrix.push_back(_iou);
    }

    return cost_matrix;
}

/**
 * @brief Calculates the IoU-based distance matrix between two sets of tracks.
 *
 * This function computes the Intersection over Union (IoU) distance between two sets of tracks based on their bounding
 * boxes (Top, Left, Bottom, Right). The IoU distance is defined as 1 - IoU, where lower IoU values indicate greater
 * distance (less overlap). The resulting cost matrix is used for removing duplicate tracks between two sets of tracks.
 *
 * @param atracks Vector of STrack objects representing the first set of tracks (existing tracks).
 * @param btracks Vector of STrack objects representing the second set of tracks ( new detections).
 *
 * @return A 2D vector representing the cost matrix, where each element [i][j] is the IoU distance (1 - IoU)
 *         between atracks[i] and btracks[j].
 */
vector <vector<float>> BYTETracker::iou_distance(vector <STrack> &atracks, vector <STrack> &btracks) {
    vector <vector<float>> atlbrs, btlbrs;
    for (int i = 0; i < atracks.size(); i++) {
        atlbrs.push_back(atracks[i].tlbr);
    }
    for (int i = 0; i < btracks.size(); i++) {
        btlbrs.push_back(btracks[i].tlbr);
    }

    vector <vector<float>> _ious = ious(atlbrs, btlbrs);
    vector <vector<float>> cost_matrix;
    for (int i = 0; i < _ious.size(); i++) {
        vector<float> _iou;
        for (int j = 0; j < _ious[i].size(); j++) {
            _iou.push_back(1 - _ious[i][j]);
        }
        cost_matrix.push_back(_iou);
    }

    return cost_matrix;
}

/**
 * @brief Solves the Linear Assignment Problem using the Jonker-Volgenant algorithm.
 *
 * Computes optimal row-to-column assignments to minimize total cost, supporting non-square matrix extension and cost limits.
 * Used in tracking to associate detections (columns) with tracks (rows).
 *
 * @param cost 2D vector of costs where cost[i][j] is the cost of assigning row i to column j.
 * @param rowsol Vector storing column indices assigned to each row (-1 for no assignment).
 * @param colsol Vector storing row indices assigned to each column (-1 for no assignment).
 * @param extend_cost If true, extends non-square matrices to square ones; else, exits on non-square input.
 * @param cost_limit Max cost for assignments; if < LONG_MAX, fills extended matrix with cost_limit / 2.0.
 * @param return_cost If true, returns total assignment cost; else, returns 0.0.
 *
 * @return Total cost of assignments if return_cost is true; else, 0.0.
 */
double BYTETracker::lapjv(const vector <vector<float>> &cost, vector<int> &rowsol, vector<int> &colsol,
                          bool extend_cost, float cost_limit, bool return_cost) {
    vector <vector<float>> cost_c;
    cost_c.assign(cost.begin(), cost.end());

    vector <vector<float>> cost_c_extended;

    int n_rows = cost.size();
    int n_cols = cost[0].size();
    rowsol.resize(n_rows);
    colsol.resize(n_cols);

    int n = 0;
    if (n_rows == n_cols) {
        n = n_rows;
    } else {
        if (!extend_cost) {
            cout << "set extend_cost=True" << endl;
            system("pause");
            exit(0);
        }
    }

    if (extend_cost || cost_limit < LONG_MAX) {
        n = n_rows + n_cols;
        cost_c_extended.resize(n);
        for (int i = 0; i < cost_c_extended.size(); i++)
            cost_c_extended[i].resize(n);

        if (cost_limit < LONG_MAX) {
            for (int i = 0; i < cost_c_extended.size(); i++) {
                for (int j = 0; j < cost_c_extended[i].size(); j++) {
                    cost_c_extended[i][j] = cost_limit / 2.0;
                }
            }
        } else {
            float cost_max = -1;
            for (int i = 0; i < cost_c.size(); i++) {
                for (int j = 0; j < cost_c[i].size(); j++) {
                    if (cost_c[i][j] > cost_max)
                        cost_max = cost_c[i][j];
                }
            }
            for (int i = 0; i < cost_c_extended.size(); i++) {
                for (int j = 0; j < cost_c_extended[i].size(); j++) {
                    cost_c_extended[i][j] = cost_max + 1;
                }
            }
        }

        for (int i = n_rows; i < cost_c_extended.size(); i++) {
            for (int j = n_cols; j < cost_c_extended[i].size(); j++) {
                cost_c_extended[i][j] = 0;
            }
        }
        for (int i = 0; i < n_rows; i++) {
            for (int j = 0; j < n_cols; j++) {
                cost_c_extended[i][j] = cost_c[i][j];
            }
        }

        cost_c.clear();
        cost_c.assign(cost_c_extended.begin(), cost_c_extended.end());
    }

    double **cost_ptr = new double *[n];
    for (int i = 0; i < n; i++)
        cost_ptr[i] = new double[n];

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cost_ptr[i][j] = cost_c[i][j];
        }
    }

    int *x_c = new int[sizeof(int) * n];
    int *y_c = new int[sizeof(int) * n];

    int ret = lapjv_internal(n, cost_ptr, x_c, y_c);
    if (ret != 0) {
        cout << "Calculate Wrong!" << endl;
        system("pause");
        exit(0);
    }

    double opt = 0.0;

    if (n != n_rows) {
        for (int i = 0; i < n; i++) {
            if (x_c[i] >= n_cols)
                x_c[i] = -1;
            if (y_c[i] >= n_rows)
                y_c[i] = -1;
        }
        for (int i = 0; i < n_rows; i++) {
            rowsol[i] = x_c[i];
        }
        for (int i = 0; i < n_cols; i++) {
            colsol[i] = y_c[i];
        }

        if (return_cost) {
            for (int i = 0; i < rowsol.size(); i++) {
                if (rowsol[i] != -1) {
                    //cout << i << "\t" << rowsol[i] << "\t" << cost_ptr[i][rowsol[i]] << endl;
                    opt += cost_ptr[i][rowsol[i]];
                }
            }
        }
    } else if (return_cost) {
        for (int i = 0; i < rowsol.size(); i++) {
            opt += cost_ptr[i][rowsol[i]];
        }
    }

    for (int i = 0; i < n; i++) {
        delete[]cost_ptr[i];
    }
    delete[]cost_ptr;
    delete[]x_c;
    delete[]y_c;

    return opt;
}

// Generates a unique RGB color for a given object identity index using modular arithmetic
Scalar BYTETracker::get_color(int idx) {
    idx += 3;
    return Scalar(37 * idx % 255, 17 * idx % 255, 29 * idx % 255);
}