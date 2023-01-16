// #include <builder/trt_builder.hpp>
// #include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
// #include <app_clpr/yolo_plate.hpp>
// #include <app_clpr/plate_rec.hpp>

// #include <opencv2/freetype.hpp>

#include <app_clpr/plate_e2e.hpp>
using namespace std;
/*
static shared_ptr<Clpr::DetInfer> get_yolo_plate(TRT::Mode mode, const string& model, int device_id) {
    auto mode_name = TRT::mode_string(mode);
    TRT::set_device(device_id);
    auto int8process = [=](int current, int count, const vector<string>& files, shared_ptr<TRT::Tensor>& tensor) {
        INFO("Int8 %d / %d", current, count);

        for (int i = 0; i < files.size(); ++i) {
            auto image = cv::imread(files[i]);
            Clpr::image_to_tensor(image, tensor, i);
        }
    };

    const char* name = model.c_str();
    INFO("===================== test %s %s ==================================", mode_name, name);
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

    return Clpr::create_det(model_file,  // engine file
                            device_id,   // gpu id
                            0.25f,       // confidence threshold
                            0.45f,       // nms threshold
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
    auto mode       = TRT::Mode::FP16;
    auto yolo_plate = get_yolo_plate(mode, name, device_id);
    if (yolo_plate == nullptr) {
        INFOE("engine create failed.");
        return;
    }
    auto plate_rec = get_plate_rec(TRT::Mode::FP32, "plate_rec_color", device_id);
    if (plate_rec == nullptr) {
        INFOE("plate rec engine create failed.");
        return;
    }

    // warm up
    auto files = iLogger::find_files("exp", "*.jpg;*.jpeg;*.png;*.gif;*.tif");
    vector<cv::Mat> images;
    for (int i = 0; i < files.size(); ++i) {
        auto image = cv::imread(files[i]);
        images.emplace_back(image);
    }

    // warmup
    vector<shared_future<Clpr::PlateRegionArray>> boxes_array;
    for (int i = 0; i < 10; ++i)
        boxes_array = yolo_plate->commits(images);
    boxes_array.back().get();
    boxes_array.clear();

    /////////////////////////////////////////////////////////
    const int ntest  = 100;
    auto begin_timer = iLogger::timestamp_now_float();

    for (int i = 0; i < ntest; ++i)
        boxes_array = yolo_plate->commits(images);

    // wait all result
    boxes_array.back().get();

    float inference_average_time = (iLogger::timestamp_now_float() - begin_timer) / ntest / images.size();
    auto mode_name               = TRT::mode_string(mode);
    INFO("plate_detect[%s] average: %.2f ms / image, FPS: %.2f", mode_name, inference_average_time,
         1000 / inference_average_time);
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
    // 测试 commits
    vector<cv::Mat> images_copy{image, image, image};
    auto results_copy = yolo_plate->commits(images_copy);
    auto results      = results_copy[0].get();
    // auto results = yolo_plate->commit(image).get();
    for (auto& r : results) {
        auto plate_result = plate_rec->commit(make_tuple(image, r.landmarks)).get();
        INFO("current plateNO is \n%s", plate_result.print_info().c_str());
        cv::rectangle(image, cv::Point(r.left, r.top), cv::Point(r.right, r.bottom), cv::Scalar(0, 255, 0), 2);
        for (int j = 0; j < 4; ++j) {
            cv::circle(image, cv::Point(r.landmarks[j * 2 + 0], r.landmarks[j * 2 + 1]), 3, cv::Scalar(0, 0, 255), -1,
                       16);
        }
        cv::putText(image, iLogger::format("%.3f", r.confidence), cv::Point(r.left, r.top), 0, 1, cv::Scalar(0, 255, 0),
                    1, 16);
        ft2->putText(image, plate_result.color_str() + plate_result.number, cv::Point(r.left, r.bottom + fontHeight),
                     fontHeight, CV_RGB(255, 255, 0), thickness, linestyle, true);
    }
    auto save_image = "det_plate.jpg";
    cv::imwrite(save_image, image);
    INFO("save image to %s.", save_image);
}
*/
static void plate_e2e_test() {
    auto e2e = Clpr::create_e2e("plate_detect-blazeface_fpn-384", "plate_rec_color", 0.4f);

    auto test_image = "exp/test1.jpg";
    cv::Mat image   = cv::imread(test_image);
    // warm up
    for (int i = 0; i < 20; ++i) {
        e2e->detect(image);
    }
    auto t1 = iLogger::timestamp_now_float();
    auto rd = e2e->detect(image);
    auto t2 = iLogger::timestamp_now_float();
    INFO("detect function cost: %fms.", float(t2 - t1));
    for (const auto& r : rd) {
        INFO("plate info is\n%s", r.print_info().c_str());
        cv::rectangle(image, r.plateRect(), cv::Scalar(0, 0, 255), 2);
    }

    // 测试find car
    Clpr::Polygon polygon1{cv::Point(141, 1105), cv::Point(877, 1134), cv::Point(1100, 533), cv::Point(550, 482)};
    Clpr::Polygon polygon2{cv::Point(874, 1134), cv::Point(1574, 1157), cv::Point(1505, 513), cv::Point(1083, 527)};
    Clpr::Polygon polygon3{cv::Point(1656, 1148), cv::Point(2291, 1141), cv::Point(1892, 495), cv::Point(1499, 504)};
    Clpr::Carports carports1{polygon2, polygon1, polygon3};
    test_image = "exp/test1.jpg";
    image      = cv::imread(test_image);

    // 耗时
    auto begin_timer = iLogger::timestamp_now_float();
    std::vector<Clpr::plateInfo> results;
    e2e->getFindCarResult(image, carports1, results);
    auto end_timer = iLogger::timestamp_now_float();
    INFO("getFindCarResult function cost %.2f ms.", (float)(end_timer - begin_timer));
    for (const auto& r : results) {
        INFO("plate info is\n%s", r.print_info().c_str());
        cv::rectangle(image, r.plateRect(), cv::Scalar(0, 0, 255), 2);
    }
    cv::polylines(image, carports1, true, (0, 255, 255));
    cv::imwrite("predict2.jpg", image);

    // 测试稳定性
    for (;;) {
        cv::Mat w_image  = cv::imread(test_image);
        auto begin_timer = iLogger::timestamp_now_float();
        std::vector<Clpr::plateInfo> w_results;
        e2e->getFindCarResult(w_image, carports1, w_results);
        auto end_timer = iLogger::timestamp_now_float();
        INFO("getFindCarResult function cost %.2f ms.", (float)(end_timer - begin_timer));
        for (const auto& r : w_results) {
            INFO("plate info is\n%s", r.print_info().c_str());
        }
    }
}
int app_plate() {
    // yolo_plate_test();
    plate_e2e_test();
    return 0;
}