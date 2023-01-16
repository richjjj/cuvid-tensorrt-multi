#include "plate_e2e.hpp"
#include "plate_rec.hpp"
#include "yolo_plate.hpp"

#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include <common/infer_controller.hpp>
#include <common/monopoly_allocator.hpp>
#include <common/preprocess_kernel.cuh>
#include <builder/trt_builder.hpp>
namespace Clpr {

static shared_ptr<DetInfer> get_yolo_plate(TRT::Mode mode, const string& model, float confidence_threshold = 0.4f,
                                           int device_id = 0) {
    auto mode_name = TRT::mode_string(mode);
    TRT::set_device(device_id);
    auto int8process = [=](int current, int count, const vector<string>& files, shared_ptr<TRT::Tensor>& tensor) {
        INFO("Int8 %d / %d", current, count);

        for (int i = 0; i < files.size(); ++i) {
            auto image = cv::imread(files[i]);
            image_to_tensor(image, tensor, i);
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

    return create_det(model_file,            // engine file
                      device_id,             // gpu id
                      confidence_threshold,  // confidence threshold
                      0.45f,                 // nms threshold
                      1024                   // max objects
    );
}

static shared_ptr<RecInfer> get_plate_rec(TRT::Mode mode, const string& model, int device_id) {
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

    return create_rec(model_file,  // engine file
                      device_id    // gpu id

    );
}

class e2eInferImpl : public e2eInfer {
public:
    virtual ~e2eInferImpl() {
        ;
    }
    bool startup(const string& det_name, const string& rec_name, float confidence_threshold = 0.4f, int gpuid = 0) {
        yolo_plate_ = get_yolo_plate(TRT::Mode::FP16, det_name, confidence_threshold, gpuid);
        if (yolo_plate_ == nullptr)
            return false;
        rec_ = get_plate_rec(TRT::Mode::FP16, rec_name, gpuid);
        if (rec_ == nullptr)
            return false;
        return true;
    }
    virtual e2eOutput detect(const e2eInput& input) override {
        e2eOutput tmp;
        auto det_result = yolo_plate_->commit(input).get();
        tmp.resize(det_result.size());
        vector<RecInput> rec_inputs;
        for (int i = 0; i < det_result.size(); i++) {
            rec_inputs.emplace_back(make_tuple(input, det_result[i].landmarks));

            tmp[i].bottom               = det_result[i].bottom;
            tmp[i].top                  = det_result[i].top;
            tmp[i].left                 = det_result[i].left;
            tmp[i].right                = det_result[i].right;
            tmp[i].plateRect_confidence = det_result[i].confidence;
            tmp[i].plateType            = det_result[i].class_label ? "multi" : "single";
        }
        auto rec_result = rec_->commits(rec_inputs);
        for (int i = 0; i < rec_result.size(); ++i) {
            auto r                       = rec_result[i].get();
            tmp[i].plateColor            = r.color_str();
            tmp[i].plateColor_confidence = r.color_confidence;
            tmp[i].plateNO               = r.number;
            tmp[i].plateNO_confidence    = r.number_confidence;
            tmp[i].carType               = "轿车";
        }
        return std::move(tmp);
    }
    virtual vector<e2eOutput> detects(const vector<e2eInput>& inputs) override {
        vector<e2eOutput> tmp_array;
        auto det_reults_array = yolo_plate_->commits(inputs);
        tmp_array.resize(det_reults_array.size());
        for (int j = 0; j < det_reults_array.size(); ++j) {
            auto det_result = det_reults_array[j].get();
            auto tmp        = tmp_array[j];
            tmp.resize(det_result.size());
            vector<RecInput> rec_inputs;
            for (int i = 0; i < det_result.size(); i++) {
                rec_inputs.emplace_back(make_tuple(inputs[j], det_result[i].landmarks));

                tmp[i].bottom               = det_result[i].bottom;
                tmp[i].top                  = det_result[i].top;
                tmp[i].left                 = det_result[i].left;
                tmp[i].right                = det_result[i].right;
                tmp[i].plateRect_confidence = det_result[i].confidence;
                tmp[i].plateType            = det_result[i].class_label ? "multi" : "single";
            }
            auto rec_result = rec_->commits(rec_inputs);
            for (int i = 0; i < rec_result.size(); ++i) {
                auto r                       = rec_result[i].get();
                tmp[i].plateColor            = r.color_str();
                tmp[i].plateColor_confidence = r.color_confidence;
                tmp[i].plateNO               = r.number;
                tmp[i].plateNO_confidence    = r.number_confidence;
                tmp[i].carType               = "轿车";
            }
        }
        return move(tmp_array);
    }
    virtual int getFindCarResult(const e2eInput& input, const Carports& carports,
                                 std::vector<plateInfo>& results) override {
        auto t0 = iLogger::timestamp_now_float();
        vector<DetInput> tmp;
        tmp.reserve(carports.size());
        for (auto& c : carports) {
            tmp.emplace_back(input, c);
        }
        results.resize(carports.size());
        auto det_reults_array = yolo_plate_->commits(tmp);
        for (int i = 0; i < det_reults_array.size(); ++i) {
            // 只取每个det carport 的第一个结果
            // DEBUG 测试耗时
            auto t1         = iLogger::timestamp_now_float();
            auto det_result = det_reults_array[i].get();
            auto t2         = iLogger::timestamp_now_float();
            if (!det_result.empty()) {
                // 只取score值最大的
                sort(det_result.begin(), det_result.end(),
                     [](auto& a, auto& b) { return a.confidence > b.confidence; });
                // 车牌识别
                auto t3                         = iLogger::timestamp_now_float();
                auto rec_output                 = rec_->commit(make_tuple(input, det_result[0].landmarks));
                auto t4                         = iLogger::timestamp_now_float();
                results[i].bottom               = det_result[0].bottom;
                results[i].top                  = det_result[0].top;
                results[i].left                 = det_result[0].left;
                results[i].right                = det_result[0].right;
                results[i].plateRect_confidence = det_result[0].confidence;
                results[i].plateType            = det_result[0].class_label ? "multi" : "single";
                results[i].carType              = "轿车";

                // 获取车牌结果
                auto r = rec_output.get();
                // 肯定有结果

                results[i].plateColor            = r.color_str();
                results[i].plateColor_confidence = r.color_confidence;
                results[i].plateNO               = r.number;
                results[i].plateNO_confidence    = r.number_confidence;
                auto t5                          = iLogger::timestamp_now_float();
                INFOD("cost---- det: %fms;copy: %fms;rec: %fms;copy: %fms", float(t2 - t1), float(t3 - t2),
                      float(t4 - t3), float(t5 - t4));
            }
        }
        auto tn = iLogger::timestamp_now_float();
        INFOD("function cost: %fms", float(tn - t0));
        return 0;
    }

private:
    shared_ptr<DetInfer> yolo_plate_;
    shared_ptr<RecInfer> rec_;
};
shared_ptr<e2eInfer> create_e2e(const string& det_name, const string& rec_name, float confidence_threshold,
                                float nms_threshold, int gpuid) {
    // 设置loger
    iLogger::set_log_level(iLogger::LogLevel::Info);
    iLogger::set_logger_save_directory("/tmp/trtpro");
    shared_ptr<e2eInferImpl> instance(new e2eInferImpl());
    if (!instance->startup(det_name, rec_name, confidence_threshold)) {
        instance.reset();
    }
    return instance;
}

}  // namespace Clpr