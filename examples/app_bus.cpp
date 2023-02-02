/*************************************************************************************
 * Description: multi hard decode with tensorrt
 * Version: 1.0
 * Company: xmrbi
 * Author: zhongchong
 * Date: 2023-02-02 09:01:58
 * LastEditors: zhongchong
 * LastEditTime: 2023-02-02 10:20:59
 *************************************************************************************/

#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include "app_yolo/yolo.hpp"
#include "app_yolo/multi_gpu.hpp"

using namespace std;

static const char* cocolabels[] = {
    "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
    "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"};

static void inference_and_performance(int deviceid, const string& engine_file, TRT::Mode mode, Yolo::Type type,
                                      const string& model_name) {
    auto engine = Yolo::create_infer(engine_file,               // engine file
                                     type,                      // yolo type, Yolo::Type::V5 / Yolo::Type::X
                                     deviceid,                  // gpu id
                                     0.25f,                     // confidence threshold
                                     0.45f,                     // nms threshold
                                     Yolo::NMSMethod::FastGPU,  // NMS method, fast GPU / CPU
                                     1024,                      // max objects
                                     false                      // preprocess use multi stream
    );
    if (engine == nullptr) {
        INFOE("Engine is nullptr");
        return;
    }

    auto files = iLogger::find_files("inference", "*.jpg;*.jpeg;*.png;*.gif;*.tif");
    vector<cv::Mat> images;
    for (int i = 0; i < files.size(); ++i) {
        auto image = cv::imread(files[i]);
        images.emplace_back(image);
    }

    // warmup
    cv::Mat w_image(640, 480, CV_8UC3);
    vector<cv::Mat> w_images(40, w_image);
    INFO("w_images.size() = %d", w_images.size());
    vector<shared_future<Yolo::BoxArray>> boxes_array;
    for (int i = 0; i < 10; ++i)
        boxes_array = engine->commits(w_images);
    boxes_array.back().get();
    boxes_array.clear();

    /////////////////////////////////////////////////////////
    const int ntest  = 100;
    auto begin_timer = iLogger::timestamp_now_float();

    for (int i = 0; i < ntest; ++i)
        boxes_array = engine->commits(images);

    // wait all result
    boxes_array.back().get();

    float inference_average_time = (iLogger::timestamp_now_float() - begin_timer) / ntest / images.size();
    auto type_name               = Yolo::type_name(type);
    auto mode_name               = TRT::mode_string(mode);
    INFO("%s[%s] average: %.2f ms / image, FPS: %.2f", engine_file.c_str(), type_name, inference_average_time,
         1000 / inference_average_time);

    cv::Mat bus_image = cv::imread("inference/bus.jpg");
    auto bus_result   = engine->commit(bus_image).get();

    auto& boxes = bus_result;
    for (auto& obj : boxes) {
        if (obj.class_label == 5) {
            uint8_t b, g, r;
            tie(b, g, r) = iLogger::random_color(obj.class_label);
            cv::rectangle(bus_image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom),
                          cv::Scalar(b, g, r), 5);

            auto name    = cocolabels[obj.class_label];
            auto caption = iLogger::format("%s %.2f", name, obj.confidence);
            int width    = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
            cv::rectangle(bus_image, cv::Point(obj.left - 3, obj.top - 33), cv::Point(obj.left + width, obj.top),
                          cv::Scalar(b, g, r), -1);
            cv::putText(bus_image, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
        }
    }

    string save_path = iLogger::format("bus_result.jpg");
    INFO("Save to %s, %d object, average time %.2f ms", save_path.c_str(), boxes.size(), inference_average_time);
    cv::imwrite(save_path, bus_image);
    engine.reset();
}

static void test(Yolo::Type type, TRT::Mode mode, const string& model) {
    int deviceid = 0;

    TRT::set_device(deviceid);
    const char* mode_name = TRT::mode_string(mode);
    const char* name      = model.c_str();
    INFO("===================== test %s %s %s ==================================", Yolo::type_name(type), mode_name,
         name);

    string onnx_file    = iLogger::format("%s.onnx", name);
    string model_file   = iLogger::format("%s.%s.trtmodel", name, mode_name);
    int test_batch_size = 16;

    if (not iLogger::exists(model_file)) {
        TRT::compile(mode,             // FP32、FP16、INT8
                     test_batch_size,  // max batch size
                     onnx_file,        // source
                     model_file,       // save to
                     {});
    }

    inference_and_performance(deviceid, model_file, mode, type, name);
}

int app_bus() {
    // multi_instances_test();
    // test(Yolo::Type::V7, TRT::Mode::FP32, "yolov7");
    test(Yolo::Type::V5, TRT::Mode::FP32, "yolov5s");
    return 0;
}