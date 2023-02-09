/*************************************************************************************
 * Description: multi hard decode with tensorrt
 * Version: 1.0
 * Company: xmrbi
 * Author: zhongchong
 * Date: 2023-01-13 13:06:23
 * LastEditors: zhongchong
 * LastEditTime: 2023-02-08 17:21:55
 *************************************************************************************/
#pragma once

#include "kalmanFilter.h"
#include <queue>

using namespace std;

template <class T>
class CircleQueue {
public:
    CircleQueue(const unsigned int length) : size_(length) {}
    void push(const T &item) {
        if (data_.size() > size_) {
            data_.pop_front();
        }
        data_.emplace_back(item);
    };

public:
    deque<T> data_;

private:
    unsigned int size_{0};
};
enum TrackState { New = 0, Tracked, Lost, Removed };

class STrack {
public:
    STrack(vector<float> tlwh_, float score, int det_index);
    ~STrack();

    vector<float> static tlbr_to_tlwh(vector<float> &tlbr);
    void static multi_predict(vector<STrack *> &stracks, byte_kalman::KalmanFilter &kalman_filter);
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
    void assign_last_current_tlbr(vector<float> new_tlbr) {
        this->datas_.push(new_tlbr);
        // this->last_tlbr    = this->current_tlbr;
        this->current_tlbr = new_tlbr;
    }

public:
    bool is_activated;
    int track_id;
    int state;
    int detection_index{};
    vector<float> _tlwh;  // tracker第一帧的坐标
    vector<float> tlwh;
    vector<float> tlbr;
    CircleQueue<vector<float>> datas_{30};
    // vector<float> last_tlbr;
    vector<float> current_tlbr;
    int frame_id;
    int tracklet_len;
    int start_frame;

    KAL_MEAN mean;
    KAL_COVA covariance;
    float score;

private:
    byte_kalman::KalmanFilter kalman_filter;
};