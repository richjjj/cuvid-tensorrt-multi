#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include "app_yolopose_gpuptr/yolo_gpuptr.hpp"

using namespace std;

static shared_ptr<YoloposeGPUPtr::Infer> get_yolo(YoloposeGPUPtr::Type type, TRT::Mode mode, const string &model,
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

static void render_to_images(vector<shared_future<YoloposeGPUPtr::BoxArray>> &objs_array, const string &name) {
    iLogger::rmtree(name);
    iLogger::mkdir(name);

    cv::VideoCapture capture("exp/35.mp4");
    cv::Mat image;

    if (!capture.isOpened()) {
        INFOE("Open video failed.");
        return;
    }

    int iframe = 0;
    while (capture.read(image) && iframe < objs_array.size()) {
        auto objs = objs_array[iframe].get();
        for (auto &obj : objs) {
            uint8_t b, g, r;
            tie(b, g, r) = iLogger::random_color(obj.class_label);
            cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r),
                          5);

            auto name    = "person";
            auto caption = iLogger::format("%s %.2f", name, obj.confidence);
            int width    = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
            cv::rectangle(image, cv::Point(obj.left - 3, obj.top - 33), cv::Point(obj.left + width, obj.top),
                          cv::Scalar(b, g, r), -1);
            cv::putText(image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
            for (int j = 0; j < 17; j++) {
                auto x = obj.pose[j * 3];
                auto y = obj.pose[j * 3 + 1];
                cv::circle(image, cv::Point(x, y), 2, cv::Scalar(255, 0, 0), 2);
            }
        }

        cv::imwrite(iLogger::format("%s/%03d.jpg", name.c_str(), iframe), image);
        iframe++;
    }
}

static void test_soft_decode() {
    auto name          = "soft";
    int yolo_device_id = 0;
    auto yolo          = get_yolo(YoloposeGPUPtr::Type::V5, TRT::Mode::FP32, "yolov5s_pose", yolo_device_id);
    if (yolo == nullptr) {
        INFOE("Yolo create failed");
        return;
    }

    cv::VideoCapture capture("exp/35.mp4");
    cv::Mat image;

    if (!capture.isOpened()) {
        INFOE("Open video failed.");
        return;
    }

    // warm up
    for (int i = 0; i < 10; ++i)
        yolo->commit(cv::Mat(640, 640, CV_8UC3)).get();

    vector<shared_future<YoloposeGPUPtr::BoxArray>> all_boxes;
    auto tic = iLogger::timestamp_now_float();
    while (capture.read(image)) {
        all_boxes.emplace_back(yolo->commit(image));
    }

    all_boxes.back().get();
    auto toc = iLogger::timestamp_now_float();
    INFO("soft decode and inference time: %.2f ms", toc - tic);

    render_to_images(all_boxes, name);
}

int app_yolopose() {
    test_soft_decode();
    return 0;
}
