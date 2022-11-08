#include "pipeline.hpp"
#include "app_yolopose_gpuptr/yolo_gpuptr.hpp"
#include "app_yolo_gpuptr/yolo_gpuptr.hpp"
#include "builder/trt_builder.hpp"
#include "common/cuda_tools.hpp"
#include "common/ilogger.hpp"
#include "common/json.hpp"
#include "ffhdd/cuvid-decoder.hpp"
#include "ffhdd/ffmpeg-demuxer.hpp"

#include "track/bytetrack/BYTETracker.h"
#include <atomic>
#include <chrono>
#include <vector>

namespace Pipeline {
template <typename T>
T round_up(T value, int decimal_places) {
    const T multiplier = std::pow(10.0, decimal_places);
    return std::ceil(value * multiplier) / multiplier;
}
vector<Object> det2tracks(const ObjectDetector::BoxArray &array) {
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
static shared_ptr<YoloGPUPtr::Infer> get_yolo(YoloGPUPtr::Type type, TRT::Mode mode, const string &model,
                                              int device_id) {
    auto mode_name = TRT::mode_string(mode);
    TRT::set_device(device_id);

    auto int8process = [=](int current, int count, const vector<string> &files, shared_ptr<TRT::Tensor> &tensor) {
        INFO("Int8 %d / %d", current, count);

        for (int i = 0; i < files.size(); ++i) {
            auto image = cv::imread(files[i]);
            YoloGPUPtr::image_to_tensor(image, tensor, type, i);
        }
    };

    const char *name = model.c_str();
    INFO("===================== test %s %s %s ==================================", YoloGPUPtr::type_name(type),
         mode_name, name);

    string onnx_file    = iLogger::format("%s.onnx", name);
    string model_file   = iLogger::format("%s.%s.trtmodel", name, mode_name);
    int test_batch_size = 16;

    if (!iLogger::exists(model_file)) {
        TRT::compile(mode,             // FP32、FP16、INT8
                     test_batch_size,  // max batch size
                     onnx_file,        // source
                     model_file,       // save to
                     {}, int8process, "inference");
    }

    return YoloGPUPtr::create_infer(model_file,  // engine file
                                    type,        // yolo type, YoloposeGPUPtr::Type::V5 / YoloposeGPUPtr::Type::X
                                    device_id,   // gpu id
                                    0.25f,       // confidence threshold
                                    0.45f,       // nms threshold
                                    YoloGPUPtr::NMSMethod::FastGPU,  // NMS method, fast GPU / CPU
                                    1024                             // max objects
    );
}
static shared_ptr<YoloposeGPUPtr::Infer> get_yolopose(YoloposeGPUPtr::Type type, TRT::Mode mode, const string &model,
                                                      int device_id) {
    auto mode_name = TRT::mode_string(mode);
    TRT::set_device(device_id);

    auto int8process = [=](int current, int count, const vector<string> &files, shared_ptr<TRT::Tensor> &tensor) {
        INFO("Int8 %d / %d", current, count);

        for (int i = 0; i < files.size(); ++i) {
            auto image = cv::imread(files[i]);
            YoloposeGPUPtr::image_to_tensor(image, tensor, type, i);
        }
    };

    const char *name = model.c_str();
    INFO("===================== test %s %s %s ==================================", YoloposeGPUPtr::type_name(type),
         mode_name, name);

    string onnx_file    = iLogger::format("%s.onnx", name);
    string model_file   = iLogger::format("%s.%s.trtmodel", name, mode_name);
    int test_batch_size = 16;

    if (!iLogger::exists(model_file)) {
        TRT::compile(mode,             // FP32、FP16、INT8
                     test_batch_size,  // max batch size
                     onnx_file,        // source
                     model_file,       // save to
                     {}, int8process, "inference");
    }

    return YoloposeGPUPtr::create_infer(model_file,  // engine file
                                        type,        // yolo type, YoloposeGPUPtr::Type::V5 / YoloposeGPUPtr::Type::X
                                        device_id,   // gpu id
                                        0.25f,       // confidence threshold
                                        0.45f,       // nms threshold
                                        YoloposeGPUPtr::NMSMethod::FastGPU,  // NMS method, fast GPU / CPU
                                        1024                                 // max objects
    );
}

class PipelineImpl : public Pipeline {
public:
    virtual ~PipelineImpl() {
        join();
    }
    virtual void join() override {
        for (auto &t : ts_) {
            if (t.joinable())
                t.join();
        }
        INFO("pipeline done.");
    }

    virtual bool make_view(const string &uri, size_t timeout) override {
        promise<bool> pro;
        ts_.emplace_back(thread(&PipelineImpl::worker, this, uri, ref(pro)));
        bool state = pro.get_future().get();
        if (state)
            uris_.emplace_back(uri);
        else
            INFOE("The uri connection is refused.");
        return state;
    }
    virtual vector<bool> make_views(const vector<string> &uris, size_t timeout) override {
        vector<bool> out;
        for (const auto &uri : uris) {
            out.emplace_back(make_view(uri, timeout));
        }
        return out;
    }
    virtual void worker(const string &uri, promise<bool> &state) {
        auto demuxer = FFHDDemuxer::create_ffmpeg_demuxer(uri, true);
        if (demuxer == nullptr) {
            INFOE("demuxer create failed");
            state.set_value(false);
            return;
        }

        auto decoder = FFHDDecoder::create_cuvid_decoder(
            use_device_frame_, FFHDDecoder::ffmpeg2NvCodecId(demuxer->get_video_codec()), -1, gpu_);

        if (decoder == nullptr) {
            INFOE("decoder create failed");
            state.set_value(false);
            return;
        }
        state.set_value(true);
        BYTETracker tracker;
        tracker.config()
            .set_initiate_state({0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 1, 0.2})
            .set_per_frame_motion({0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 1, 0.2})
            .set_max_time_lost(150);

        uint8_t *packet_data = nullptr;
        int packet_size      = 0;
        uint64_t pts         = 0;

        // demuxer->get_extra_data(&packet_data, &packet_size);
        // decoder->decode(packet_data, packet_size);
        do {
            bool flag = demuxer->demux(&packet_data, &packet_size, &pts);
            if (!flag) {
                INFOW("diconnected will be reopened.");
                while (!flag) {
                    this_thread::sleep_for(chrono::milliseconds(200));
                    flag    = demuxer->reopen();
                    decoder = FFHDDecoder::create_cuvid_decoder(
                        use_device_frame_, FFHDDecoder::ffmpeg2NvCodecId(demuxer->get_video_codec()), -1, gpu_);
                }
                INFOW("reopen successed.");
            }
            INFO("current uri is %s", uri.c_str());
            int ndecoded_frame = decoder->decode(packet_data, packet_size, pts);
            for (int i = 0; i < ndecoded_frame; ++i) {
                unsigned int frame_index = 0;
                if (callback_) {
                    nlohmann::json tmp_json;
                    tmp_json["cameraId"]     = uri;
                    tmp_json["freshTime"]    = pts;  // 时间戳，表示当前的帧数
                    tmp_json["det_results"]  = nlohmann::json::array();
                    tmp_json["pose_results"] = nlohmann::json::array();
                    tmp_json["gcn_results"]  = nlohmann::json::array();
                    YoloGPUPtr::Image image(decoder->get_frame(&pts, &frame_index), decoder->get_width(),
                                            decoder->get_height(), gpu_, decoder->get_stream(),
                                            YoloGPUPtr::ImageType::GPUBGR);
                    auto objs_future = yolo_->commit(image);
                    if (yolo_pose_ != nullptr) {
                        // YoloposeGPUPtr::Image poseimage(decoder->get_frame(&pts, &frame_index), decoder->get_width(),
                        //                                 decoder->get_height(), gpu_, decoder->get_stream(),
                        //                                 YoloposeGPUPtr::ImageType::GPUBGR);
                        auto objs_pose = yolo_pose_->commit(image).get();
                        for (const auto &obj_pose : objs_pose) {
                            vector<float> pose(obj_pose.pose, obj_pose.pose + 51);
                            nlohmann::json event_json = {
                                {"box", {obj_pose.left, obj_pose.top, obj_pose.right, obj_pose.bottom}},
                                {"pose", pose},
                                {"score", obj_pose.confidence}};
                            tmp_json["pose_results"].emplace_back(event_json);
                        }
                    }
                    auto objs = objs_future.get();
                    for (const auto &obj : objs) {
                        nlohmann::json event_json = {{"id", -1},
                                                     {"box", {obj.left, obj.top, obj.right, obj.bottom}},
                                                     {"class_label", obj.class_label},
                                                     {"score", obj.confidence}};
                        tmp_json["det_results"].emplace_back(event_json);
                    }
                    cv::Mat cvimage(image.get_height(), image.get_width(), CV_8UC3);
                    cudaMemcpyAsync(cvimage.data, image.device_data, image.get_data_size(), cudaMemcpyDeviceToHost,
                                    decoder->get_stream());
                    cudaStreamSynchronize(decoder->get_stream());
                    // cv::imwrite(cv::format("imgs/%03d.jpg", frame_index), cvimage);
                    callback_(2, (void *)&cvimage, (char *)tmp_json.dump().c_str(), tmp_json.dump().size());
                }
            }

        } while (true);
        INFO("done %s", uri.c_str());
    }
    virtual void disconnect_views(const vector<string> &dis_uris) override {}
    virtual void set_callback(ai_callback callback) override {
        callback_ = callback;
    }
    virtual void get_uris(vector<string> &current_uris) const override {
        for (const auto &x : uris_)
            current_uris.emplace_back(x);
    }
    virtual bool startup(const string &det_name, const string &pose_name, const string &gcn_name, int gpuid,
                         bool use_device_frame) {
        gpu_              = gpuid;
        use_device_frame_ = use_device_frame_;
        if (!pose_name.empty()) {
            yolo_pose_ = get_yolopose(YoloposeGPUPtr::Type::V5, TRT::Mode::FP32, pose_name, gpuid);
            if (yolo_pose_ != nullptr) {
                INFO("yolo_pose will be committed");
                for (int i = 0; i < 10; ++i)
                    yolo_pose_->commit(cv::Mat(640, 640, CV_8UC3)).get();
            }
        }
        yolo_ = get_yolo(YoloGPUPtr::Type::V5, TRT::Mode::FP16, det_name, 0);
        if (yolo_ == nullptr) {
            INFOE("create tensorrt engine failed.");
            return false;
        }
        // use_device_frame_ = use_device_frame_;
        // gpu_ = gpu_;
        for (int i = 0; i < 10; ++i)
            yolo_->commit(cv::Mat(640, 640, CV_8UC3)).get();

        return true;
    }

private:
    int gpu_               = 0;
    bool use_device_frame_ = true;
    shared_ptr<YoloposeGPUPtr::Infer> yolo_pose_;
    shared_ptr<YoloGPUPtr::Infer> yolo_;
    vector<thread> ts_;
    vector<string> uris_{};
    ai_callback callback_;
};  // namespace Pipeline
shared_ptr<Pipeline> create_pipeline(const string &det_name, const string &pose_name, const string &gcn_name, int gpuid,
                                     bool use_device_frame) {
    shared_ptr<PipelineImpl> instance(new PipelineImpl());
    if (!instance->startup(det_name, pose_name, gcn_name, gpuid, use_device_frame)) {
        instance.reset();
    }
    return instance;
}
}  // namespace Pipeline