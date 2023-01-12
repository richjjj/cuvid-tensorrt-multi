
#include <opencv2/opencv.hpp>
#include <ffhdd/simple-logger.hpp>
#include <ffhdd/ffmpeg-demuxer.hpp>
#include <ffhdd/cuvid-decoder.hpp>
#include <ffhdd/nalu.hpp>
#include <queue>
#include <mutex>
#include <tuple>
#include <future>
#include <condition_variable>
#include <common/cuda_tools.hpp>
#include <ffhdd/multi-camera.hpp>
#include "app_yolopose_gpuptr/yolo_gpuptr.hpp"
#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>

using namespace std;

static const char *cocolabels[] = {
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

bool requires(const char *name);

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

    if (!requires(name))
        return nullptr;

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

static void test_multi_decode() {
    iLogger::rmtree("imgs");
    iLogger::mkdir("imgs");

    int yolo_device_id = 0;
    auto yolo          = get_yolo(YoloposeGPUPtr::Type::V5, TRT::Mode::FP32, "yolov5s_pose", yolo_device_id);
    if (yolo == nullptr) {
        INFOE("Yolo create failed");
        return;
    }

    auto decoder = FFHDMultiCamera::create_decoder(true, -1, yolo_device_id);
    vector<thread> ts;
    int ids[64]   = {0};
    auto callback = [&](FFHDMultiCamera::View *pview, uint8_t *pimage_data, int device_id, int width, int height,
                        FFHDDecoder::FrameType type, uint64_t timestamp, FFHDDecoder::ICUStream stream) {
        unsigned int frame_index = 0;
        YoloposeGPUPtr::Image image(pimage_data, width, height, device_id, stream, YoloposeGPUPtr::ImageType::GPUBGR);
        auto objs = yolo->commit(image).get();

        cv::Mat cvimage(height, width, CV_8UC3);
        cudaMemcpyAsync(cvimage.data, pimage_data, width * height * 3, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        for (auto &obj : objs) {
            uint8_t b, g, r;
            tie(b, g, r) = iLogger::random_color(obj.class_label);
            cv::rectangle(cvimage, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r),
                          5);

            auto name    = cocolabels[obj.class_label];
            auto caption = iLogger::format("%s %.2f", name, obj.confidence);
            int width    = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
            cv::rectangle(cvimage, cv::Point(obj.left - 3, obj.top - 33), cv::Point(obj.left + width, obj.top),
                          cv::Scalar(b, g, r), -1);
            cv::putText(cvimage, caption, cv::Point(obj.left, obj.top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
        }

        cv::imwrite(cv::format("imgs/%02d_%03d.jpg", pview->get_idd(), ++ids[pview->get_idd()]), cvimage);
        std::cout << "get_width:" << width << ",get_height:" << height << ",id:" << pview->get_idd() << std::endl;
        // std::cout << "current idd is " << pview->get_idd() << std::endl;
    };

    auto func = [&](shared_ptr<FFHDMultiCamera::View> view) {
        if (view == nullptr) {
            INFOE("View is nullptr");
            return;
        }

        view->set_callback(callback);
        while (view->demux()) {
            // 模拟真实视频流
            this_thread::sleep_for(chrono::milliseconds(30));
        }
        INFO("Done> %d", view->get_idd());
    };

    for (int i = 0; i < 64; ++i) {
        if (i % 3 == 0)
            ts.emplace_back(std::bind(func, decoder->make_view("exp/dog.mp4")));
        else if (i % 3 == 1)
            ts.emplace_back(std::bind(func, decoder->make_view("exp/cat.mp4")));
        else if (i % 3 == 2)
            ts.emplace_back(std::bind(func, decoder->make_view("exp/pig.mp4")));
    }
    std::vector<std::string> current_uris;
    for (auto &t : ts)
        t.join();
    decoder->join();
    INFO("Program done.");
}

int app_multi_camera() {
    test_multi_decode();
    return 0;
}