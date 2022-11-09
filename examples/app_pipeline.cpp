#include "app_decode_inference_pipeline/pipeline.hpp"
#include "common/ilogger.hpp"
#include <iostream>
#include "common/json.hpp"
#include <math.h>

using namespace std;
static float iou(const std::vector<float> &a, const std::vector<float> &b) {
    float cleft   = max(a[0], b[0]);
    float ctop    = max(a[1], b[1]);
    float cright  = min(a[2], b[2]);
    float cbottom = min(a[3], b[3]);

    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if (c_area == 0.0f)
        return 0.0f;

    float a_area = max(0.0f, a[2] - a[0]) * max(0.0f, a[3] - a[1]);
    float b_area = max(0.0f, b[2] - b[0]) * max(0.0f, b[3] - b[1]);
    return c_area / (a_area + b_area - c_area);
}

void callback(int callbackType, void *img, char *data, int datalen) {
    // det_results : class_label    ["head", "_","smoking","body"]
    // pose_results: 51     17 * 3 (x,y,confidence)
    // 解析json
    auto results = nlohmann::json::parse(data);

    std::cout << "results is :" << results << "\n";
}
void test_pipeline() {
    // iLogger::rmtree("imgs");
    // iLogger::mkdir("imgs");
    std::string det_name  = "yolov5x-aqm";
    std::string pose_name = "yolov5s_pose";
    std::string gcn_name  = "";
    // std::vector<std::string> uris{"exp/39.mp4", "exp/37.mp4", "exp/38.mp4",
    //                               "exp/37.mp4", "exp/38.mp4", "rtsp://192.168.170.109:554/live/streamperson"};
    std::vector<std::string> uris{"rtsp://192.168.170.109:554/live/streamperson"};

    auto pipeline = Pipeline::create_pipeline(det_name, pose_name, gcn_name);
    std::vector<std::string> current_uris{};

    if (pipeline == nullptr) {
        int *a;
        std::cout << "pipeline create failed" << std::endl;
        return;
    }
    //
    pipeline->set_callback(callback);
    auto out = pipeline->make_views(uris);
    for (auto x : out) {
        std::cout << x << std::endl;
    }

    pipeline->get_uris(current_uris);
    for (auto u : current_uris) {
        std::cout << u << std::endl;
    }

    pipeline->join();
}

int app_pipeline() {
    test_pipeline();
    return 0;
}