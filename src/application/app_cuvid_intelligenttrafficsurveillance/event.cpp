/*************************************************************************************
 * Description: multi hard decode with tensorrt
 * Version: 1.0
 * Company: xmrbi
 * Author: zhongchong
 * Date: 2023-02-06 15:24:48
 * LastEditors: zhongchong
 * LastEditTime: 2023-02-15 10:08:48
 *************************************************************************************/

#include "event.hpp"
#include "common/json.hpp"
#include <thread>
#include <atomic>
#include <future>
#include <memory>
#include <queue>
#include <mutex>
#include <condition_variable>
#include "common/cuda_tools.hpp"
#include "common/ilogger.hpp"
#include "utils.hpp"
#include "track/bytetrack/BYTETracker.h"

namespace Intelligent {

using namespace std;
using json = nlohmann::json;
ViewConfig parse_json_data(const string &raw_data) {
    ViewConfig tmp;
    json structed_data = json::parse(raw_data);
    tmp.cameraID       = structed_data["cameraID"];
    tmp.uri            = structed_data["uri"];
    for (auto &e : structed_data["events"]) {
        EventConfig tmp_event;
        tmp_event.enable    = e["enable"];
        tmp_event.eventName = e["eventName"];
        for (auto &roi : e["rois"]) {
            RoiConfig tmp_roi;
            tmp_roi.roiName   = roi["roiName"];
            tmp_roi.pointsNum = roi["pointsNum"];
            for (int i = 1; i < tmp_roi.pointsNum + 1; ++i) {
                string p = to_string(i);
                tmp_roi.points.emplace_back(roi["points"]["x" + p], roi["points"]["y" + p]);
            }
            tmp_event.rois.emplace_back(tmp_roi);
        }
        tmp.events.emplace_back(tmp_event);
    }
    return tmp;
}
vector<Object> det2tracks(const YoloGPUPtr::BoxArray &array) {
    vector<Object> outputs;
    for (int i = 0; i < array.size(); ++i) {
        auto &abox = array[i];
        Object obox;
        obox.prob    = abox.confidence;
        obox.label   = abox.class_label;
        obox.rect[0] = abox.left;
        obox.rect[1] = abox.top;
        obox.rect[2] = abox.right - abox.left;
        obox.rect[3] = abox.bottom - abox.top;
        outputs.emplace_back(obox);
    }
    return outputs;
}
unique_ptr<BYTETracker> creatTracker() {
    unique_ptr<BYTETracker> tracker(new BYTETracker());
    tracker->config()
        .set_initiate_state({0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 1, 0.2})
        .set_per_frame_motion({0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 1, 0.2})
        .set_max_time_lost(150);
    return tracker;
}
// 识别不同的事件
// 1. 从raw——data获取所有的事件类型和roi
// 2. 同时识别所有的事件
// 3. 需要保存连续若干帧的检测结果，得到目标的轨迹
// 4. 识别结果保存到tmp_json里
// 综上，构造一个Event类实现上述功能
// TODO 判断属于哪一个roi
class EventInferImpl : public EventInfer {
public:
    virtual ~EventInferImpl() {
        running_ = false;
        cv_.notify_one();
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
        // INFO("EventInfer destroy.");
    }
    virtual string get_uri() const override {
        return config_.uri;
    }
    virtual void set_callback(ai_callback callback) override {
        callback_ = callback;
    }
    virtual bool commit(const Input &input) override {
        unique_lock<mutex> l(lock_);
        cv_.wait(l, [&]() { return jobs_.size() < 20; });
        jobs_.emplace(input);
        cv_.notify_all();
        return true;
    }
    virtual bool startup(const string &raw_data) {
        // 设置callback

        // 解析config
        config_ = parse_json_data(raw_data);
        // 初始化 tracker
        tracker_       = creatTracker();
        running_       = true;
        worker_thread_ = thread(&EventInferImpl::worker, this);
        return true;
    }
    virtual void worker() {
        Input job;
        while (running_) {
            {
                unique_lock<mutex> l(lock_);
                cv_.wait(l, [&]() { return !running_ || !jobs_.empty(); });  // return false 一直等着
                if (!running_)
                    break;

                if (!jobs_.empty()) {
                    job = std::move(jobs_.front());
                    jobs_.pop();
                    cv_.notify_all();
                }
            }
            auto size = jobs_.size();
            // 业务代码
            if (job.frame_index_ != 0) {
                nlohmann::json tmp_json;
                tmp_json["cameraID"] = config_.cameraID;
                tmp_json["uri"]      = config_.uri;
                json events_json     = json::array();
                auto t1              = iLogger::timestamp_now_float();
                auto &image          = job.image;

                // auto objs        = job.future_boxarray_.get();
                auto &objs  = job.boxarray_;
                auto tracks = tracker_->update(det2tracks(objs));
                auto t2     = iLogger::timestamp_now_float();
                for (auto &e : config_.events) {
                    if (e.enable) {
                        if (e.eventName == "weiting") {
                            json objects_json = json::array();
                            for (size_t t = 0; t < tracks.size(); t++) {
                                auto &track = tracks[t];
                                auto &obj   = objs[track.detection_index];
                                // car
                                if (obj.class_label == 2 && obj.confidence > 0.4) {
                                    // 判断在哪个roi
                                    for (auto &roi : e.rois) {
                                        if (isPointInPolygon(roi.points, track.current_center_point_)) {
                                            // 判断是否停住
                                            bool stop = isStopped(track.center_points_.data_);
                                            if (stop) {
                                                json object_json = {
                                                    {"objectID", track.track_id},
                                                    {"label", 0},
                                                    {"coordinate", {obj.left, obj.top, obj.right, obj.bottom}},
                                                    {"confidence", obj.confidence},
                                                    {"roi_name", roi.roiName}};
                                                objects_json.emplace_back(object_json);
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                            if (!objects_json.empty()) {
                                json event_json = {{"eventName", "weiting"}, {"objects", objects_json}};
                                events_json.emplace_back(event_json);
                            }
                        } else if (e.eventName == "yongdu") {
                            json objects_json = json::array();
                            vector<int> car_count(e.rois.size(), 0);
                            for (size_t t = 0; t < tracks.size(); t++) {
                                auto &track = tracks[t];
                                auto &obj   = objs[track.detection_index];
                                // car
                                if (obj.class_label == 2) {
                                    for (int i = 0; i < e.rois.size(); ++i) {
                                        if (isPointInPolygon(e.rois[i].points, track.current_center_point_)) {
                                            ++car_count[i];
                                        }
                                    }
                                }
                            }
                            // 判断是否拥堵
                            for (int i = 0; i < e.rois.size(); ++i) {
                                bool jam = (car_count[i] >= 20);  // magic number
                                // 判断在哪个roi
                                if (jam) {
                                    json object_json = {
                                        {"roi", e.rois[i].roiName},
                                    };
                                    objects_json.emplace_back(object_json);
                                }
                            }
                            if (!objects_json.empty()) {
                                json event_json = {{"eventName", "yongdu"}, {"objects", objects_json}};
                                events_json.emplace_back(event_json);
                            }
                        } else if (e.eventName == "biandao") {
                            json objects_json = json::array();
                            for (size_t t = 0; t < tracks.size(); t++) {
                                auto &track = tracks[t];
                                auto &obj   = objs[track.detection_index];
                                // car
                                if (obj.class_label == 2 && obj.confidence > 0.4) {
                                    for (auto &roi : e.rois) {
                                        // bool changeLine
                                        Line l{roi.points[0], roi.points[1]};
                                        bool changeLine = isIntersect(l, track.center_points_.data_);
                                        // 判断在哪个roi
                                        if (changeLine) {
                                            json object_json = {
                                                {"objectID", track.track_id},
                                                {"label", 0},
                                                {"coordinate", {obj.left, obj.top, obj.right, obj.bottom}},
                                                {"confidence", obj.confidence},
                                                {"roi_name", roi.roiName}};
                                            objects_json.emplace_back(object_json);
                                            break;
                                        }
                                    }
                                }
                            }
                            if (!objects_json.empty()) {
                                json event_json = {{"eventName", "biandao"}, {"objects", objects_json}};
                                events_json.emplace_back(event_json);
                            }
                        } else if (e.eventName == "xingrenchuangru") {
                            json objects_json = json::array();
                            for (size_t t = 0; t < tracks.size(); t++) {
                                auto &track = tracks[t];
                                auto &obj   = objs[track.detection_index];
                                // person
                                if (obj.class_label == 0 && obj.confidence > 0.2) {
                                    for (auto &roi : e.rois) {
                                        // 1. 判断行人是否在区域里 2. 需要有位移
                                        if (isPointInPolygon(roi.points, track.current_center_point_)) {
                                            json object_json = {
                                                {"objectID", track.track_id},
                                                {"label", 1},
                                                {"coordinate", {obj.left, obj.top, obj.right, obj.bottom}},
                                                {"confidence", obj.confidence},
                                                {"roi_name", roi.roiName}};
                                            objects_json.emplace_back(object_json);
                                            break;
                                        }
                                    }
                                }
                            }
                            if (!objects_json.empty()) {
                                json event_json = {{"eventName", "xingrenchuangru"}, {"objects", objects_json}};
                                events_json.emplace_back(event_json);
                            }
                        } else if (e.eventName == "test") {
                            json objects_json = json::array();
                            for (size_t t = 0; t < tracks.size(); t++) {
                                auto &track = tracks[t];
                                auto &obj   = objs[track.detection_index];
                                // car
                                if (obj.class_label == 2 && obj.confidence > 0.2) {
                                    for (auto &roi : e.rois) {
                                        // 判断是否停住
                                        if (isPointInPolygon(roi.points, track.current_center_point_)) {
                                            json object_json = {
                                                {"objectID", track.track_id},
                                                {"label", 1},
                                                {"coordinate", {obj.left, obj.top, obj.right, obj.bottom}},
                                                {"confidence", obj.confidence},
                                                {"roi_name", roi.roiName}};
                                            objects_json.emplace_back(object_json);
                                            break;
                                        }
                                    }
                                }
                            }
                            if (!objects_json.empty()) {
                                json event_json = {{"eventName", "test"}, {"objects", objects_json}};
                                events_json.emplace_back(event_json);
                            }
                        }
                    }
                }
                auto t3 = iLogger::timestamp_now_float();

                bool isPicture = false;
                cv::Mat cvimage(image.get_height(), image.get_width(), CV_8UC3);
                if (job.frame_index_ % 50 == 0 || !events_json.empty()) {
                    cudaMemcpyAsync(cvimage.data, image.device_data, image.get_data_size(), cudaMemcpyDeviceToHost,
                                    image.stream);
                    cudaStreamSynchronize(image.stream);
                    isPicture = true;
                }
                // debug
                // if (isPicture)
                //     cv::imwrite(cv::format("imgs/%d.jpg", job.frame_index_), cvimage);
                auto t4               = iLogger::timestamp_now_float();
                tmp_json["isPicture"] = isPicture;
                tmp_json["events"]    = events_json;
                auto data             = tmp_json.dump();
                bool isEmpty          = events_json.empty();
                // if (!isEmpty)
                //     INFO("data is %s.", data.c_str());
                if (isPicture)
                    callback_(2, (void *)&cvimage, (char *)data.c_str(), data.size());
                else
                    callback_(2, nullptr, (char *)data.c_str(), data.size());
                auto t5 = iLogger::timestamp_now_float();
                // INFO("total: %.2fms; image copy: %.2f ms; track: %.2f, event: %.2f; callback: %.2f", float(t5 - t1),
                //      float(t4 - t3), float(t2 - t1), float(t3 - t2), float(t5 - t4));
                // reset
                job.frame_index_ = 0;
            }
        }

        ;
    }

private:
    atomic<bool> running_{false};
    thread worker_thread_;
    queue<Input> jobs_;
    mutex lock_;
    condition_variable cv_;
    unique_ptr<BYTETracker> tracker_;
    ViewConfig config_;
    ai_callback callback_;
};
std::shared_ptr<EventInfer> create_event(const string &raw_data) {
    shared_ptr<EventInferImpl> instance(new EventInferImpl());
    if (!instance->startup(raw_data)) {
        instance.reset();
    }
    return instance;
}
};  // namespace Intelligent