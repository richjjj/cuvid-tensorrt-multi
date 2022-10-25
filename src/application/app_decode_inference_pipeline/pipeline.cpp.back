#include "pipeline.hpp"
#include "app_yolo_gpuptr/yolo_gpuptr.hpp"
#include "builder/trt_builder.hpp"
#include "common/cuda_tools.hpp"
#include "common/ilogger.hpp"
#include "common/json.hpp"
#include "ffhdd/multi-camera.hpp"
#include "track/bytetrack/BYTETracker.h"
#include <atomic>
#include <map>
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
            decoder_->join();
            INFO("pipeline done.");
        }
        virtual void join() override
        {
            for (auto &t : ts_)
                t.join();
            decoder_->join();
            INFO("pipeline done.");
        }

        virtual void yolo_infer(FFHDMultiCamera::View *pview,
                                uint8_t *pimage_data, int device_id, int width, int height,
                                FFHDDecoder::FrameType type, uint64_t timestamp,
                                FFHDDecoder::ICUStream stream)
        {
            unsigned int frame_index = 0;
            YoloGPUPtr::Image image(
                pimage_data,
                width, height,
                device_id,
                stream,
                YoloGPUPtr::ImageType::GPUBGR);
            auto objs = yolo_pose_->commit(image).get();

            // ObjectDetector::BoxArray objs;
            nlohmann::json tmp_json;
            // int current_id = pview->get_idd();
            string current_name = pview->get_name();
            tmp_json["cameraId"] = current_name;
            tmp_json["freshTime"] = timestamp; // 时间戳，表示当前的帧数
            tmp_json["events"] = nlohmann::json::array();
            // 有人就保存
            // TODO, 训练摔倒、等GCN分类识别模型
            auto tracks = trackers_[current_name]->update(det2tracks(objs));
            // INFO("objs.size() : %d; tracks.size() : %d", objs.size(), tracks.size());
            for (int i = 0; i < tracks.size(); i++)
            {
                auto track = tracks[i];
                // INFO("objs[i] left is %f; track[i] left is %f; _tlwh[i] is %f", objs[i].left, track.tlwh[0], track._tlwh[0]);
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
                // uint8_t b, g, r;
                // tie(b, g, r) = iLogger::random_color(obj.class_label);
                // cv::rectangle(cvimage, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 5);
                // auto caption = iLogger::format("%s %.2f", "person", obj.confidence);
                // int width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
                // cv::rectangle(cvimage, cv::Point(obj.left - 3, obj.top - 33), cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
                // cv::putText(cvimage, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
            }
            // cv::imwrite(cv::format("imgs/%02d_%03d.jpg", pview->get_idd(), ++ids[pview->get_idd()]), cvimage);
            cv::Mat cvimage(height, width, CV_8UC3);
            cudaMemcpyAsync(cvimage.data, pimage_data, width * height * 3, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            if (callback_)
            {
                callback_(2, (void *)&cvimage, (char *)tmp_json.dump().c_str(), tmp_json.dump().size());
            }
        }
        virtual void make_views(const vector<string> &uris) override
        {
            auto func = [&](shared_ptr<FFHDMultiCamera::View> view)
            {
                if (view == nullptr)
                {
                    INFOE("View is nullptr");
                    return;
                }
                auto call_yolo_infer = [&](FFHDMultiCamera::View *pview,
                                           uint8_t *pimage_data, int device_id, int width, int height,
                                           FFHDDecoder::FrameType type, uint64_t timestamp,
                                           FFHDDecoder::ICUStream stream)
                {
                    this->yolo_infer(pview, pimage_data, device_id, width, height,
                                     type, timestamp,
                                     stream);
                };
                view->set_callback(call_yolo_infer);
                while (view->demux())
                {
                    // 模拟真实视频流
                    this_thread::sleep_for(chrono::milliseconds(30));
                }
                INFO("Done> %d", view->get_idd());
            };

            for (const auto &uri : uris)
            {
                uris_.emplace_back(uri);
                // BYTETracker tracker;
                trackers_[uri] = make_shared<BYTETracker>();
                trackers_[uri]->config().set_initiate_state({0.1, 0.1, 0.1, 0.1,
                                                             0.2, 0.2, 1, 0.2})
                    .set_per_frame_motion({0.1, 0.1, 0.1, 0.1,
                                           0.2, 0.2, 1, 0.2})
                    .set_max_time_lost(150);
                ts_.emplace_back(bind(func, decoder_->make_view(uri)));
            }
        }
        virtual void disconnect_views(const vector<string> &dis_uris) override
        {
            // 先停掉所有的views
            stop_signal_ = true;
            for (auto &t : ts_)
                t.join();
            ts_.clear();
            decoder_.reset();
            decoder_ = FFHDMultiCamera::create_decoder(use_device_frame_, -1, gpu_);
            // 重新启动views
            vector<string> new_uris;
            for (auto &s : uris_)
            {
                if (find(dis_uris.begin(), dis_uris.end(), s) == dis_uris.end())
                    new_uris.emplace_back(s);
            }
            uris_.clear();
            make_views(new_uris);
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
            yolo_pose_ = get_yolo(YoloGPUPtr::Type::V5, TRT::Mode::FP32, engile_file, gpuid);
            // yolo_pose_ = YoloGPUPtr::create_infer(engile_file, YoloGPUPtr::Type::V5, gpuid);
            if (yolo_pose_ == nullptr)
            {
                INFOE("create tensorrt engine failed.");
                return false;
            }
            decoder_ = FFHDMultiCamera::create_decoder(use_device_frame, -1, gpuid);
            if (decoder_ == nullptr)
            {
                INFO("create decoder failed.");
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
        shared_ptr<FFHDMultiCamera::Decoder> decoder_;
        map<string, shared_ptr<BYTETracker>> trackers_;
        vector<thread> ts_;
        vector<string> uris_{};
        atomic<bool> stop_signal_{false};
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
};