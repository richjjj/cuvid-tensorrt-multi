#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include <app_clpr/yolo_plate.hpp>

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