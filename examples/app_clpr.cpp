#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include <app_clpr/yolo_plate.hpp>
#include <app_clpr/plate_rec.hpp>

#include <opencv2/freetype.hpp>

using namespace std;

static shared_ptr<Clpr::DetInfer> get_yolo_plate(TRT::Mode mode, const string& model, int device_id) {
    auto mode_name = TRT::mode_string(mode);
    TRT::set_device(device_id);

    const char* name = model.c_str();
    INFO("===================== test %s %s ==================================", mode_name, name);
    string onnx_file    = iLogger::format("%s.onnx", name);
    string model_file   = iLogger::format("%s.%s.trtmodel", name, mode_name);
    int test_batch_size = 16;

    if (!iLogger::exists(model_file)) {
        TRT::compile(mode,             // FP32、FP16、INT8
                     test_batch_size,  // max batch size
                     onnx_file,        // source
                     model_file        // save to
        );
    }

    return Clpr::create_det(model_file,  // engine file
                            device_id,   // gpu id
                            0.25f,       // confidence threshold
                            0.45f,       // nms threshold
                                         // NMS method, fast GPU / CPU
                            1024         // max objects
    );
}

static shared_ptr<Clpr::RecInfer> get_plate_rec(TRT::Mode mode, const string& model, int device_id) {
    auto mode_name = TRT::mode_string(mode);
    TRT::set_device(device_id);

    const char* name = model.c_str();
    INFO("===================== test %s %s ==================================", mode_name, name);
    string onnx_file    = iLogger::format("%s.onnx", name);
    string model_file   = iLogger::format("%s.%s.trtmodel", name, mode_name);
    int test_batch_size = 16;

    if (!iLogger::exists(model_file)) {
        TRT::compile(mode,             // FP32、FP16、INT8
                     test_batch_size,  // max batch size
                     onnx_file,        // source
                     model_file        // save to
        );
    }

    return Clpr::create_rec(model_file,  // engine file
                            device_id    // gpu id

    );
}

static void yolo_plate_test() {
    auto name       = "plate_detect";
    auto device_id  = 0;
    auto yolo_plate = get_yolo_plate(TRT::Mode::FP32, name, device_id);
    if (yolo_plate == nullptr) {
        INFOE("engine create failed.");
        return;
    }
    auto plate_rec = get_plate_rec(TRT::Mode::FP32, "plate_rec", device_id);
    if (plate_rec == nullptr) {
        INFOE("plate rec engine create failed.");
        return;
    }

    // warm up
    for (int i = 0; i < 10; ++i) {
        yolo_plate->commit(cv::Mat(640, 640, CV_8UC3)).get();
    }

    // inference
    auto ft2 = cv::freetype::createFreeType2();
    ft2->loadFontData("simfang.ttf", 0);
    int fontHeight  = 30;
    int thickness   = -1;
    int linestyle   = 8;
    auto test_image = "exp/plate.jpg";
    cv::Mat image   = cv::imread(test_image);
    auto results    = yolo_plate->commit(image).get();
    for (auto& r : results) {
        auto plateno = plate_rec->commit(make_tuple(image, r.landmarks)).get();
        INFO("current plateNO is %s", plateno.c_str());
        cv::rectangle(image, cv::Point(r.left, r.top), cv::Point(r.right, r.bottom), cv::Scalar(0, 255, 0), 2);
        for (int j = 0; j < 4; ++j) {
            cv::circle(image, cv::Point(r.landmarks[j * 2 + 0], r.landmarks[j * 2 + 1]), 3, cv::Scalar(0, 0, 255), -1,
                       16);
        }
        cv::putText(image, iLogger::format("%.3f", r.confidence), cv::Point(r.left, r.top), 0, 1, cv::Scalar(0, 255, 0),
                    1, 16);
        ft2->putText(image, plateno, cv::Point(r.left, r.bottom + fontHeight), fontHeight, CV_RGB(255, 255, 0),
                     thickness, linestyle, true);
    }
    auto save_image = "det_plate.jpg";
    cv::imwrite(save_image, image);
}

int app_plate() {
    yolo_plate_test();
    return 0;
}