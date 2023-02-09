/*************************************************************************************
 * Description: multi hard decode with tensorrt
 * Version: 1.0
 * Company: xmrbi
 * Author: zhongchong
 * Date: 2023-02-01 10:12:40
 * LastEditors: zhongchong
 * LastEditTime: 2023-02-09 09:29:57
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

namespace Intelligent {
using json = nlohmann::json;
using namespace std;

static json parse_raw_data(const string &raw_data) {
    return std::move(json::parse(raw_data));
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
        json j_data    = json::parse(raw_data);
        string uri     = j_data["uri"];
        runnings_[uri] = true;
        // 需要指定GPU device
        ts_.emplace_back(thread(&IntelligentTrafficImpl::worker, this, uri, j_data, ref(pro)));
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
    virtual void worker(const string &uri, json json_data, promise<bool> &state) {
        auto demuxer = FFHDDemuxer::create_ffmpeg_demuxer(uri, true);
        if (demuxer == nullptr) {
            INFOE("demuxer create failed");
            state.set_value(false);
            return;
        }
        auto gpu_id = devices_[get_gpu_index()];
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
                    auto t1 = iLogger::timestamp_now_float();
                    YoloGPUPtr::Image image(decoder->get_frame(&pts, &frame_index), decoder->get_width(),
                                            decoder->get_height(), gpu_id, decoder->get_stream(),
                                            YoloGPUPtr::ImageType::GPUBGR);
                    nlohmann::json tmp_json;
                    tmp_json["cameraID"] = json_data["cameraID"];
                    tmp_json["uri"]      = uri;
                    json events_json     = json::array();

                    auto objs_future = infers_[gpu_id]->commit(image);

                    cv::Mat cvimage(image.get_height(), image.get_width(), CV_8UC3);
                    cudaMemcpyAsync(cvimage.data, image.device_data, image.get_data_size(), cudaMemcpyDeviceToHost,
                                    decoder->get_stream());
                    cudaStreamSynchronize(decoder->get_stream());
                    // 识别不同的事件
                    // 1. 从raw——data获取所有的事件类型和roi
                    // 2. 同时识别所有的事件
                    // 3. 需要保存连续若干帧的检测结果，得到目标的轨迹
                    // 4. 识别结果保存到tmp_json里
                    // 综上，构造一个Event类实现上述功能
                    auto objs   = objs_future.get();
                    auto tracks = tracker->update(det2tracks(objs));

                    for (auto &e : json_data["events"]) {
                        if (e["enable"]) {
                            if (e["EventName"] == "weiting") {
                                json objects_json = json::array();
                                for (size_t t = 0; t < tracks.size(); t++) {
                                    auto &track = tracks[t];
                                    // car
                                    if (objs[t].class_label == 2) {
                                        // 判断是否停住
                                        bool stop =
                                            (abs(track.datas_.data_.front()[2] - track.datas_.data_.back()[2]) < 1);
                                        // 判断在哪个roi
                                        if (stop) {
                                            json object_json = {{"objectID", track.track_id},
                                                                {"label", 0},
                                                                {"coordinate", track.datas_.data_.front()},
                                                                {"roi_name", "roi_name"}};
                                            objects_json.emplace_back(object_json);
                                        }
                                    }
                                }
                                if (!objects_json.empty()) {
                                    json event_json = {{"eventName", "weiting"}, {"objects", objects_json}};
                                    events_json.emplace_back(event_json);
                                }
                            } else if (e["EventName"] == "yongdu") {
                                json objects_json = json::array();
                                int car_count     = 0;
                                for (size_t t = 0; t < tracks.size(); t++) {
                                    auto &track = tracks[t];
                                    // car
                                    if (objs[t].class_label == 2) {
                                        car_count++;
                                    }
                                }
                                // 判断是否拥堵
                                bool jam = (car_count >= 20);  // magic number
                                // 判断在哪个roi
                                if (jam) {
                                    json object_json = {
                                        {"roi", "roi_name"},
                                    };
                                    objects_json.emplace_back(object_json);
                                }
                                if (!objects_json.empty()) {
                                    json event_json = {
                                        {"eventName", "yongdu"}, {"label", 0}, {"objects", objects_json}};
                                    events_json.emplace_back(event_json);
                                }
                            } else if (e["EventName"] == "biandao") {
                                json objects_json = json::array();
                                for (size_t t = 0; t < tracks.size(); t++) {
                                    auto &track = tracks[t];
                                    // car
                                    if (objs[t].class_label == 2) {
                                        bool stop =
                                            (abs(track.datas_.data_.front()[2] - track.datas_.data_.back()[2]) < 20);
                                        // 判断在哪个roi
                                        if (stop) {
                                            json object_json = {{"objectID", track.track_id},
                                                                {"label", 0},
                                                                {"coordinate", track.datas_.data_.front()}};
                                            objects_json.emplace_back(object_json);
                                        }
                                    }
                                }
                                if (!objects_json.empty()) {
                                    json event_json = {{"eventName", "biandao"}, {"objects", objects_json}};
                                    events_json.emplace_back(event_json);
                                }
                            } else if (e["EventName"] == "xingrenchuangru") {
                                json objects_json = json::array();
                                for (size_t t = 0; t < tracks.size(); t++) {
                                    auto &track = tracks[t];
                                    // person
                                    if (objs[t].class_label == 0) {
                                        // 判断是否停住
                                        bool stop =
                                            (abs(track.datas_.data_.front()[2] - track.datas_.data_.back()[2]) < 20);
                                        // 判断在哪个roi
                                        if (stop) {
                                            json object_json = {{"objectID", track.track_id},
                                                                {"label", 1},
                                                                {"coordinate", track.datas_.data_.front()}};
                                            objects_json.emplace_back(object_json);
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
                    auto data          = tmp_json.dump();
                    callback_(2, (void *)&cvimage, (char *)data.c_str(), data.size());
                    float d2h_time = iLogger::timestamp_now_float() - t1;
                    INFO("image copy from device to host cost %.2f ms.", d2h_time);
                }
            }
        }
        INFO("done %s", uri.c_str());
    }
    int get_gpu_index() {
        return ((cursor_++) + 1) % infers_.size();
    }
    virtual void set_callback(ai_callback callback) override {
        callback_ = callback;
    }
    virtual vector<string> get_uris() const override {
        return uris_;
    }
    virtual bool startup(const string &model_repository, const vector<int> gpuids) {
        model_repository_ = model_repository;
        devices_          = gpuids;
        // load model
        // infers_.resize(gpuids.size());
#pragma omp parallel for num_threads(gpuids.size())
        for (auto &gpuid : gpuids) {
            infers_[gpuid] =
                YoloGPUPtr::create_infer(model_repository + "/yolov8n.FP16.trtmodel", YoloGPUPtr::Type::V5, gpuid);
            for (int i = 0; i < 20; ++i) {
                // warm up
                infers_[gpuid]->commit(cv::Mat(640, 640, CV_8UC3)).get();
            }
            INFO("infers_[%d] warm done.", gpuid);
        }
        for (auto &gpuid : gpuids) {
            if (infers_[gpuid] == nullptr) {
                INFOE("Infer create failed, gpuid = %d", gpuid);
                return false;
            }
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
    map<unsigned int, shared_ptr<YoloGPUPtr::Infer>> infers_;
    atomic<unsigned int> cursor_{0};
};
shared_ptr<IntelligentTraffic> create_intelligent_traffic(const string &model_repository, const vector<int> gpuids) {
    iLogger::set_logger_save_directory("/tmp/intelligent_traffic");
    shared_ptr<IntelligentTrafficImpl> instance(new IntelligentTrafficImpl());
    if (!instance->startup(model_repository, gpuids)) {
        instance.reset();
    }
    return instance;
}

};  // namespace Intelligent