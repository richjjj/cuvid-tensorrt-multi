/*************************************************************************************
 * Description: multi hard decode with tensorrt
 * Version: 1.0
 * Company: xmrbi
 * Author: zhongchong
 * Date: 2023-02-01 10:12:40
 * LastEditors: zhongchong
 * LastEditTime: 2023-02-13 16:05:02
 *************************************************************************************/
#include "intelligent_traffic.hpp"
#include "track/bytetrack/BYTETracker.h"
#include "app_yolo_gpuptr/yolo_gpuptr.hpp"

#include "builder/trt_builder.hpp"
#include "common/cuda_tools.hpp"
#include "common/ilogger.hpp"
#include "common/json.hpp"
#include "ffhdd/cuvid-decoder.hpp"
#include "ffhdd/ffmpeg-demuxer.hpp"
#include "utils.hpp"
#include "event.hpp"

namespace Intelligent {
using json = nlohmann::json;
using namespace std;

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
class IntelligentTrafficImpl : public IntelligentTraffic {
public:
    IntelligentTrafficImpl() {}
    virtual bool make_view(const string &raw_data, size_t timeout) override {
        promise<bool> pro;
        // parse data 得到接口文档的结果
        auto config    = parse_json_data(raw_data);
        string uri     = config.uri;
        runnings_[uri] = true;
        // 需要指定GPU device
        ts_.emplace_back(thread(&IntelligentTrafficImpl::worker, this, uri, config, ref(pro)));
        bool state = pro.get_future().get();
        if (state) {
            uris_.emplace_back(uri);
        } else {
            INFOE("The uri connection is refused.");
            runnings_[uri] = false;
        }
        return state;
    }
    unique_ptr<BYTETracker> creatTracker() {
        unique_ptr<BYTETracker> tracker(new BYTETracker());
        tracker->config()
            .set_initiate_state({0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 1, 0.2})
            .set_per_frame_motion({0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 1, 0.2})
            .set_max_time_lost(150);
        return tracker;
    }
    virtual void worker(const string &uri, const ViewConfig &config, promise<bool> &state) {
        auto demuxer = FFHDDemuxer::create_ffmpeg_demuxer(uri, true);
        if (demuxer == nullptr) {
            INFOE("demuxer create failed");
            state.set_value(false);
            return;
        }

        auto thread_id = ++thread_id_;
        auto gpu_id    = devices_[get_gpu_index()];

        int instance_id = ((device_count_map_[gpu_id]++) + 1) % instances_per_device_;
        // debug
        // INFO("current gpu_id is %d", gpu_id);
        auto decoder = FFHDDecoder::create_cuvid_decoder(
            true, FFHDDecoder::ffmpeg2NvCodecId(demuxer->get_video_codec()), -1, gpu_id);
        if (decoder == nullptr) {
            INFOE("decoder create failed");
            state.set_value(false);
            return;
        }
        state.set_value(true);
        auto tracker = creatTracker();

        if (tracker == nullptr) {
            INFOE("tracker create failed");
            state.set_value(false);
            return;
        }
        uint8_t *packet_data = nullptr;
        int packet_size      = 0;
        uint64_t pts         = 0;
        while (runnings_[uri]) {
            bool flag = demuxer->demux(&packet_data, &packet_size, &pts);
            if (!flag && runnings_[uri]) {
                while (!flag && runnings_[uri]) {
                    INFO("%s cannot be connected. try reconnect....", uri.c_str());
                    iLogger::sleep(200);
                    demuxer.reset();
                    demuxer = FFHDDemuxer::create_ffmpeg_demuxer(uri, true);
                    flag    = (demuxer != nullptr) && demuxer->demux(&packet_data, &packet_size, &pts);
                }
                if (!runnings_[uri]) {
                    INFO("disconnect %s", uri.c_str());
                } else {
                    INFO("%s reopen successed.", uri.c_str());
                }
            }
            // INFO("current uri is %s", uri.c_str());
            int ndecoded_frame = decoder->decode(packet_data, packet_size, pts);
            if (ndecoded_frame == -1) {
                INFO("%s stopped.", uri.c_str());
                runnings_[uri] = false;
            }
            for (int i = 0; i < ndecoded_frame; ++i) {
                unsigned int frame_index = 0;
                if (callback_) {
                    YoloGPUPtr::Image image(decoder->get_frame(&pts, &frame_index), decoder->get_width(),
                                            decoder->get_height(), gpu_id, decoder->get_stream(),
                                            YoloGPUPtr::ImageType::GPUBGR);
                    nlohmann::json tmp_json;
                    tmp_json["cameraID"] = config.cameraID;
                    tmp_json["uri"]      = uri;
                    json events_json     = json::array();
                    auto t1              = iLogger::timestamp_now_float();
                    auto objs_future     = infers_[gpu_id][instance_id]->commit(image);

                    cv::Mat cvimage(image.get_height(), image.get_width(), CV_8UC3);

                    // 识别不同的事件
                    // 1. 从raw——data获取所有的事件类型和roi
                    // 2. 同时识别所有的事件
                    // 3. 需要保存连续若干帧的检测结果，得到目标的轨迹
                    // 4. 识别结果保存到tmp_json里
                    // 综上，构造一个Event类实现上述功能
                    // TODO 判断属于哪一个roi
                    auto objs        = objs_future.get();
                    float infer_time = iLogger::timestamp_now_float() - t1;
                    // INFO("image inference cost %.2f ms.", infer_time);
                    auto tracks      = tracker->update(det2tracks(objs));
                    float track_time = iLogger::timestamp_now_float() - t1;
                    // INFO("image inference and track cost %.2f ms.", track_time);

                    for (auto &e : config.events) {
                        if (e.enable) {
                            if (e.eventName == "weiting") {
                                json objects_json = json::array();
                                for (size_t t = 0; t < tracks.size(); t++) {
                                    auto &track = tracks[t];
                                    auto &obj   = objs[track.detection_index];
                                    // car
                                    if (obj.class_label == 2 && obj.confidence > 0.5) {
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
                                    json event_json = {
                                        {"eventName", "yongdu"}, {"label", 0}, {"objects", objects_json}};
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
                                    if (obj.class_label == 0 && obj.confidence > 0.5) {
                                        for (auto &roi : e.rois) {
                                            // 判断是否停住
                                            if (isPointInPolygon(roi.points, track.current_center_point_)) {
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
                                    json event_json = {{"eventName", "xingrenchuangru"}, {"objects", objects_json}};
                                    events_json.emplace_back(event_json);
                                }
                            }
                        }
                    }
                    tmp_json["events"] = events_json;

                    bool isPicture = false;
                    if (frame_index % 50 == 0 || !events_json.empty()) {
                        cudaMemcpyAsync(cvimage.data, image.device_data, image.get_data_size(), cudaMemcpyDeviceToHost,
                                        decoder->get_stream());
                        cudaStreamSynchronize(decoder->get_stream());
                        isPicture = true;
                    }
                    tmp_json["isPicture"] = isPicture;
                    auto data             = tmp_json.dump();
                    float d2h_time        = iLogger::timestamp_now_float() - t1;
                    callback_(2, (void *)&cvimage, (char *)data.c_str(), data.size());
                    float call_time = iLogger::timestamp_now_float() - t1;
                    INFO("[%d]  [%d]--[%d] infer:%.2fms; track:%.2fms; e2e: %.2fms. call_back: %.2fms", thread_id,
                         gpu_id, instance_id, infer_time, track_time, d2h_time, call_time);
                }
            }
        }
        INFO("done %s", uri.c_str());
    }
    int get_gpu_index() {
        // auto index = ((cursor_++) + 1) % infers_.size();
        return ((cursor_++) + 1) % infers_.size();
    }
    virtual void set_callback(ai_callback callback) override {
        callback_ = callback;
    }
    virtual vector<string> get_uris() const override {
        return uris_;
    }
    virtual bool startup(const string &model_repository, const vector<int> gpuids, int instances_per_device) {
        model_repository_     = model_repository;
        devices_              = gpuids;
        instances_per_device_ = instances_per_device;

        // #pragma omp parallel for num_threads(devices_.size())
        for (auto &gpuid : devices_) {
            // 每个GPU多个个instances，当下设置为2个
            for (int i = 0; i < instances_per_device_; ++i) {
                infers_[gpuid].emplace_back(std::move(YoloGPUPtr::create_infer(
                    model_repository + "/yolov8n.FP16.trtmodel", YoloGPUPtr::Type::V5, gpuid)));
            }
            INFO("instance.size()=%d", infers_[gpuid].size());
            for (int i = 0; i < 20; ++i) {
                // warm up
                for (auto &infer : infers_[gpuid]) {
                    infer->commit(cv::Mat(640, 640, CV_8UC3)).get();
                }
            }
            INFO("infers_[%d] warm done.", gpuid);
        }
        for (auto &gpuid : devices_) {
            for (auto &infer : infers_[gpuid]) {
                if (infer == nullptr) {
                    INFO("Infer create failed, gpuid = %d", gpuid);
                    return false;
                }
            }
            device_count_map_[gpuid] = 0;
        }
        return true;
    }

    virtual void join() {
        for (auto &t : ts_) {
            if (t.joinable())
                t.join();
        }
    }

    virtual void stop() override {
        for (auto &r : uris_) {
            runnings_[r] = false;
        }
    }
    virtual void disconnect_view(const string &dis_uri) override {
        runnings_[dis_uri] = false;
    }
    virtual ~IntelligentTrafficImpl() {
        join();
        INFO("Traffic done.");
    }

private:
    // multi gpus
    vector<int> devices_{0};
    vector<thread> ts_;
    vector<string> uris_;
    map<string, atomic_bool> runnings_;
    ai_callback callback_;
    string model_repository_;
    // vector<shared_ptr<YoloGPUPtr::Infer>> infers_;
    int instances_per_device_{1};
    map<unsigned int, vector<shared_ptr<YoloGPUPtr::Infer>>> infers_;
    // map<int, atomic<unsigned int>> device_count_map_;  //{gpuid:数目}
    atomic<int> device_count_map_[4];
    atomic<unsigned int> thread_id_{0};
    atomic<unsigned int> cursor_{0};
};
shared_ptr<IntelligentTraffic> create_intelligent_traffic(const string &model_repository, const vector<int> gpuids,
                                                          int instances_per_device) {
    iLogger::set_logger_save_directory("/tmp/intelligent_traffic");
    shared_ptr<IntelligentTrafficImpl> instance(new IntelligentTrafficImpl());
    if (!instance->startup(model_repository, gpuids, instances_per_device)) {
        instance.reset();
    }
    return instance;
}
};  // namespace Intelligent