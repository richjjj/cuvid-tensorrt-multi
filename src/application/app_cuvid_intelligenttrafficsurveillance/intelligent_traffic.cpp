/*************************************************************************************
 * Description: multi hard decode with tensorrt
 * Version: 1.0
 * Company: xmrbi
 * Author: zhongchong
 * Date: 2023-02-01 10:12:40
 * LastEditors: zhongchong
 * LastEditTime: 2023-02-03 16:56:59
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

class IntelligentTrafficImpl : public IntelligentTraffic {
public:
    IntelligentTrafficImpl() {}
    virtual bool make_view(const string &raw_data, size_t timeout) override {
        promise<bool> pro;
        // parse data 得到接口文档的结果
        auto j_data    = json::parse(raw_data);
        string uri     = j_data["uri"];
        runnings_[uri] = true;
        // 需要指定GPU device
        ts_.emplace_back(thread(&IntelligentTrafficImpl::worker, this, uri, ref(pro)));
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
    virtual void worker(const string &uri, promise<bool> &state) {
        auto demuxer = FFHDDemuxer::create_ffmpeg_demuxer(uri, true);
        if (demuxer == nullptr) {
            INFOE("demuxer create failed");
            state.set_value(false);
            return;
        }
        auto gpu_id = devices_[get_gpu_index()];
        INFO("current gpu_id is %d", gpu_id);
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
            if (!flag) {
                while (!flag && runnings_[uri]) {
                    INFOW("%s cannot be connected. try reconnect....", uri.c_str());
                    this_thread::sleep_for(chrono::milliseconds(200));
                    demuxer.reset();
                    demuxer = FFHDDemuxer::create_ffmpeg_demuxer(uri, true);
                    flag    = (demuxer != nullptr) && demuxer->demux(&packet_data, &packet_size, &pts);
                }
                if (!runnings_[uri]) {
                    INFO("disconnect %s", uri.c_str());
                } else {
                    INFOW("%s reopen successed.", uri.c_str());
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
                if (true) {
                    YoloGPUPtr::Image image(decoder->get_frame(&pts, &frame_index), decoder->get_width(),
                                            decoder->get_height(), gpu_id, decoder->get_stream(),
                                            YoloGPUPtr::ImageType::GPUBGR);
                    nlohmann::json tmp_json;
                    tmp_json["cameraId"]    = uri;
                    tmp_json["freshTime"]   = frame_index;  // 时间戳，表示当前的帧数
                    tmp_json["det_results"] = nlohmann::json::array();
                    auto t1                 = iLogger::timestamp_now_float();
                    auto objs_future        = infers_[gpu_id]->commit(image);

                    // cv::Mat cvimage(image.get_height(), image.get_width(), CV_8UC3);
                    // cudaMemcpyAsync(cvimage.data, image.device_data, image.get_data_size(), cudaMemcpyDeviceToHost,
                    //                 decoder->get_stream());
                    // cudaStreamSynchronize(decoder->get_stream());

                    auto objs      = objs_future.get();
                    float d2h_time = iLogger::timestamp_now_float() - t1;
                    INFO("image copy from device to host cost %.2f ms.", d2h_time);
                    for (const auto &obj : objs) {
                        nlohmann::json event_json = {{"box", {obj.left, obj.top, obj.right, obj.bottom}},
                                                     {"class_label", obj.class_label},
                                                     {"score", obj.confidence}};
                        tmp_json["det_results"].emplace_back(event_json);
                    }

                    // callback_(2, (void *)&cvimage, (char *)tmp_json.dump().c_str(), tmp_json.dump().size());
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