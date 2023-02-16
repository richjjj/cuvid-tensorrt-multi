/*************************************************************************************
 * Description: multi hard decode with tensorrt
 * Version: 1.0
 * Company: xmrbi
 * Author: zhongchong
 * Date: 2023-02-01 10:12:40
 * LastEditors: zhongchong
 * LastEditTime: 2023-02-15 11:17:03
 *************************************************************************************/
#include "intelligent_traffic.hpp"
#include "app_yolo_gpuptr/yolo_gpuptr.hpp"

#include "common/cuda_tools.hpp"
#include "common/ilogger.hpp"
#include "ffhdd/cuvid-decoder.hpp"
#include "ffhdd/ffmpeg-demuxer.hpp"
#include "event.hpp"

namespace Intelligent {
using namespace std;

class IntelligentTrafficImpl : public IntelligentTraffic {
public:
    IntelligentTrafficImpl() {}
    virtual bool make_view(const string &raw_data, size_t timeout) override {
        if (callback_ == nullptr) {
            INFOE("please set_callback befor make_view.");
            return false;
        }
        promise<bool> pro;
        // 创建eventinfer对象
        auto event_infer = create_event(raw_data);
        event_infer->set_callback(callback_);
        string uri     = event_infer->get_uri();
        runnings_[uri] = true;
        ts_.emplace_back(thread(&IntelligentTrafficImpl::worker, this, uri, event_infer, ref(pro)));
        bool state = pro.get_future().get();
        if (state) {
            uris_.emplace_back(uri);
        } else {
            // INFOE("The uri connection is refused.");
            event_infer.reset();
            runnings_[uri] = false;
        }
        return state;
    }
    virtual void worker(const string &uri, shared_ptr<EventInfer> event_infer, promise<bool> &state) {
        auto demuxer = FFHDDemuxer::create_ffmpeg_demuxer(uri, true);
        if (demuxer == nullptr) {
            INFOE("demuxer create failed");
            state.set_value(false);
            return;
        }
        // create decode
        auto gpu_id     = devices_[get_gpu_index()];
        int instance_id = ((device_count_map_[gpu_id]++) + 1) % instances_per_device_;
        auto decoder    = FFHDDecoder::create_cuvid_decoder(
            true, FFHDDecoder::ffmpeg2NvCodecId(demuxer->get_video_codec()), -1, gpu_id);
        if (decoder == nullptr) {
            INFOE("decoder create failed");
            state.set_value(false);
            cursor_--;
            device_count_map_[gpu_id]--;
            return;
        }
        state.set_value(true);
        auto thread_id = ++thread_id_;

        // decode部分
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
            auto t0            = iLogger::timestamp_now_float();
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
                    auto t1          = iLogger::timestamp_now_float();
                    auto objs_future = infers_[gpu_id][instance_id]->commit(image);

                    auto objs = objs_future.get();
                    auto t2   = iLogger::timestamp_now_float();
                    event_infer->commit({frame_index, image, objs});
                    auto t3 = iLogger::timestamp_now_float();
                    INFO("[%d]  [%d]--[%d] decode: %.2f; infer: %.2f; commit: %.2f", thread_id, gpu_id, instance_id,
                         float(t1 - t0), (float)(t2 - t1), (float)(t3 - t2));
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
                    model_repository + "/yolov6n.FP16.B32.trtmodel", YoloGPUPtr::Type::V5, gpuid)));
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
        join();
    }
    virtual void disconnect_view(const string &dis_uri) override {
        runnings_[dis_uri] = false;
    }
    virtual ~IntelligentTrafficImpl() {
        stop();
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