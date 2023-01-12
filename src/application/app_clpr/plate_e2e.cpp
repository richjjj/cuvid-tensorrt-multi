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

static shared_ptr<DetInfer> get_yolo_plate(TRT::Mode mode, const string& model, int device_id) {
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

    return create_det(model_file,  // engine file
                      device_id,   // gpu id
                      0.25f,       // confidence threshold
                      0.45f,       // nms threshold
                      1024         // max objects
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
    bool startup(const string& det_name, const string& rec_name, int gpuid = 0) {
        yolo_plate_ = get_yolo_plate(TRT::Mode::FP16, det_name, gpuid);
        if (yolo_plate_ == nullptr)
            return false;
        rec_ = get_plate_rec(TRT::Mode::FP32, rec_name, gpuid);
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

            tmp[i].bottom         = det_result[i].bottom;
            tmp[i].top            = det_result[i].top;
            tmp[i].left           = det_result[i].left;
            tmp[i].right          = det_result[i].right;
            tmp[i].box_confidence = det_result[i].confidence;
            tmp[i].plate_type     = det_result[i].class_label ? "multi" : "single";
        }
        auto rec_result = rec_->commits(rec_inputs);
        for (int i = 0; i < rec_result.size(); ++i) {
            auto r                   = rec_result[i].get();
            tmp[i].color             = r.color_str();
            tmp[i].color_confidence  = r.color_confidence;
            tmp[i].number            = r.number;
            tmp[i].number_confidence = r.number_confidence;
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

                tmp[i].bottom         = det_result[i].bottom;
                tmp[i].top            = det_result[i].top;
                tmp[i].left           = det_result[i].left;
                tmp[i].right          = det_result[i].right;
                tmp[i].box_confidence = det_result[i].confidence;
                tmp[i].plate_type     = det_result[i].class_label ? "multi" : "single";
            }
            auto rec_result = rec_->commits(rec_inputs);
            for (int i = 0; i < rec_result.size(); ++i) {
                auto r                   = rec_result[i].get();
                tmp[i].color             = r.color_str();
                tmp[i].color_confidence  = r.color_confidence;
                tmp[i].number            = r.number;
                tmp[i].number_confidence = r.number_confidence;
            }
        }
        return move(tmp_array);
    }

private:
    shared_ptr<DetInfer> yolo_plate_;
    shared_ptr<RecInfer> rec_;
};
shared_ptr<e2eInfer> create_e2e(const string& det_name, const string& rec_name, float confidence_threshold,
                                float nms_threshold, int gpuid) {
    shared_ptr<e2eInferImpl> instance(new e2eInferImpl());
    if (!instance->startup(det_name, rec_name)) {
        instance.reset();
    }
    return instance;
}

}  // namespace Clpr