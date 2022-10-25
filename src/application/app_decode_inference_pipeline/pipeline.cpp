#include "pipeline.hpp"
#include "app_yolo_gpuptr/yolo_gpuptr.hpp"
#include "builder/trt_builder.hpp"
#include "common/cuda_tools.hpp"
#include "common/ilogger.hpp"
#include "common/json.hpp"
#include "ffhdd/cuvid-decoder.hpp"
#include "ffhdd/ffmpeg-demuxer.hpp"

#include "track/bytetrack/BYTETracker.h"
#include <atomic>
#include <vector>

namespace Pipeline
{
    vector<Object> det2tracks(const ObjectDetector::BoxArray &array)
    {

        vector<Object> outputs;
        for (int i = 0; i < array.size(); ++i)
        {
            auto &abox = array[i];
            Object obox;
            obox.prob = abox.confidence;
            obox.label = abox.class_label;
            obox.rect[0] = abox.left;
            obox.rect[1] = abox.top;
            obox.rect[2] = abox.right - abox.left;
            obox.rect[3] = abox.bottom - abox.top;
            outputs.emplace_back(obox);
        }
        return outputs;
    }
    static shared_ptr<YoloGPUPtr::Infer> get_yolo(YoloGPUPtr::Type type, TRT::Mode mode, const string &model, int device_id)
    {

        auto mode_name = TRT::mode_string(mode);
        TRT::set_device(device_id);

        auto int8process = [=](int current, int count, const vector<string> &files, shared_ptr<TRT::Tensor> &tensor)
        {
            INFO("Int8 %d / %d", current, count);

            for (int i = 0; i < files.size(); ++i)
            {
                auto image = cv::imread(files[i]);
                YoloGPUPtr::image_to_tensor(image, tensor, type, i);
            }
        };

        const char *name = model.c_str();
        INFO("===================== test %s %s %s ==================================", YoloGPUPtr::type_name(type), mode_name, name);

        string onnx_file = iLogger::format("%s.onnx", name);
        string model_file = iLogger::format("%s.%s.trtmodel", name, mode_name);
        int test_batch_size = 16;

        if (!iLogger::exists(model_file))
        {
            TRT::compile(
                mode,            // FP32、FP16、INT8
                test_batch_size, // max batch size
                onnx_file,       // source
                model_file,      // save to
                {},
                int8process,
                "inference");
        }

        return YoloGPUPtr::create_infer(
            model_file,                     // engine file
            type,                           // yolo type, YoloGPUPtr::Type::V5 / YoloGPUPtr::Type::X
            device_id,                      // gpu id
            0.25f,                          // confidence threshold
            0.45f,                          // nms threshold
            YoloGPUPtr::NMSMethod::FastGPU, // NMS method, fast GPU / CPU
            1024                            // max objects
        );
    }

    class PipelineImpl : public Pipeline
    {
    public:
        virtual ~PipelineImpl()
        {
            for (auto &t : ts_)
                t.join();
            // decoder_->join();
            INFO("pipeline done.");
        }
        virtual void join() override
        {
            for (auto &t : ts_)
                t.join();
            // decoder_->join();
            INFO("pipeline done.");
        }
        virtual void make_views(const vector<string> &uris) override
        {
            for (const auto &uri : uris)
            {
                uris_.emplace_back(uri);
                ts_.emplace_back(thread(&PipelineImpl::worker, this, uri));
            }
        }
        virtual void worker(const string &uri)
        {
            auto demuxer = FFHDDemuxer::create_ffmpeg_demuxer(uri, true);
            if (demuxer == nullptr)
            {
                INFOE("demuxer create failed");
                return;
            }

            auto decoder = FFHDDecoder::create_cuvid_decoder(
                use_device_frame_, FFHDDecoder::ffmpeg2NvCodecId(demuxer->get_video_codec()), -1, gpu_);

            if (decoder == nullptr)
            {
                INFOE("decoder create failed");
                return;
            }
            BYTETracker tracker;
            tracker.config().set_initiate_state({0.1, 0.1, 0.1, 0.1,
                                                 0.2, 0.2, 1, 0.2})
                .set_per_frame_motion({0.1, 0.1, 0.1, 0.1,
                                       0.2, 0.2, 1, 0.2})
                .set_max_time_lost(150);

            uint8_t *packet_data = nullptr;
            int packet_size = 0;
            uint64_t pts = 0;

            demuxer->get_extra_data(&packet_data, &packet_size);
            decoder->decode(packet_data, packet_size);
            do
            {
                bool flag = demuxer->demux(&packet_data, &packet_size, &pts);
                if (!flag)
                {
                    INFOW("diconnected will be reopened.");
                }
                INFO("current uri is %s", uri.c_str());
                int ndecoded_frame = decoder->decode(packet_data, packet_size, pts);
                for (int i = 0; i < ndecoded_frame; ++i)
                {
                    unsigned int frame_index = 0;
                    YoloGPUPtr::Image image(
                        decoder->get_frame(&pts, &frame_index),
                        decoder->get_width(), decoder->get_height(),
                        gpu_,
                        decoder->get_stream(),
                        YoloGPUPtr::ImageType::GPUBGR);
                    auto objs = yolo_pose_->commit(image).get();
                    frame_index = frame_index + 1;
                    if (callback_)
                    {
                        nlohmann::json tmp_json;
                        // int current_id = pview->get_idd();
                        tmp_json["cameraId"] = uri;
                        tmp_json["freshTime"] = pts; // 时间戳，表示当前的帧数
                        tmp_json["events"] = nlohmann::json::array();
                        // TODO, 训练摔倒、等GCN分类识别模型
                        auto tracks = tracker.update(det2tracks(objs));
                        for (int i = 0; i < tracks.size(); i++)
                        {
                            auto track = tracks[i];
                            string event_string = "";
                            // 分类模型
                            if (track.tlwh[2] > track.tlwh[3])
                            {
                                event_string = "falldown";
                            }
                            vector<float> pose(objs[i].pose, objs[i].pose + 51);
                            // 手高于肩
                            // xyzxyz
                            if ((pose[9 * 3 + 1] < pose[5 * 3 + 1]) && (pose[10 * 3 + 1] < pose[6 * 3 + 1]))
                            {
                                event_string = "pickup";
                            }
                            nlohmann::json event_json = {
                                {"id", track.track_id},
                                {"event", event_string},
                                {"box", {track.tlwh[0], track.tlwh[1], track.tlwh[2] + track.tlwh[0], track.tlwh[3] + track.tlwh[1]}},
                                {"pose", pose},
                                {"entertime", ""},
                                {"outtime", ""},
                                {"score", track.score}};

                            tmp_json["events"].emplace_back(event_json);
                        }
                        cv::Mat cvimage(image.get_height(), image.get_width(), CV_8UC3);
                        cudaMemcpyAsync(cvimage.data, image.device_data, image.get_data_size(), cudaMemcpyDeviceToHost, decoder->get_stream());
                        cudaStreamSynchronize(decoder->get_stream());

                        callback_(2, (void *)&cvimage, (char *)tmp_json.dump().c_str(), tmp_json.dump().size());
                    }
                }
            } while (true > 0);
            INFO("done %s", uri.c_str());
        }
        virtual void disconnect_views(const vector<string> &dis_uris) override
        {
        }
        virtual void set_callback(ai_callback callback) override
        {
            callback_ = callback;
        }
        virtual void get_uris(vector<string> &current_uris) const override
        {
            for (const auto &x : uris_)
                current_uris.emplace_back(x);
        }
        virtual bool startup(const string &engile_file, int gpuid, bool use_device_frame)
        {
            gpu_ = gpuid;
            use_device_frame_ = use_device_frame_;
            yolo_pose_ = get_yolo(YoloGPUPtr::Type::V5, TRT::Mode::FP32, engile_file, gpuid);
            if (yolo_pose_ == nullptr)
            {
                INFOE("create tensorrt engine failed.");
                return false;
            }
            // use_device_frame_ = use_device_frame_;
            // gpu_ = gpu_;
            for (int i = 0; i < 10; ++i)
                yolo_pose_->commit(cv::Mat(640, 640, CV_8UC3)).get();

            return true;
        }

    private:
        int gpu_ = 0;
        bool use_device_frame_ = true;
        shared_ptr<YoloGPUPtr::Infer> yolo_pose_;
        vector<thread> ts_;
        vector<string> uris_{};
        ai_callback callback_;
    };
    shared_ptr<Pipeline> create_pipeline(const string &engile_file, int gpuid, bool use_device_frame)
    {
        shared_ptr<PipelineImpl> instance(new PipelineImpl());
        if (!instance->startup(engile_file, gpuid, use_device_frame))
        {
            instance.reset();
        }
        return instance;
    }
}