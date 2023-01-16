#include "app_cuvid_cangku/pipeline.hpp"
#include "common/ilogger.hpp"
#include <iostream>
#include "common/json.hpp"
#include <math.h>
#include <initializer_list>
#include "opencv2/opencv.hpp"

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
template <class T>
static float iou(initializer_list<T> &a, initializer_list<T> &b) {
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
    auto results      = nlohmann::json::parse(data);
    auto det_results  = results["det_results"];
    auto pose_results = results["pose_results"];
    auto camera_id    = results["cameraId"];
    // debug
    // cv::Mat image;
    // auto img_tmp = (cv::Mat *)img;
    // img_tmp->copyTo(image);
    auto frame_index = results["freshTime"];
    // debug
    // 判断抽烟
    for (auto &dr : det_results) {
        if (dr["class_label"] == 2) {
            // 与pose_results iou 匹配
            for (size_t i = 0; i < pose_results.size(); i++) {
                float c_iou = iou(dr["box"], pose_results[i]["box"]);
                if (c_iou > 0.8) {
                    std::string event = "smoking";
                    break;
                }
            }
        }
    }
    // 判断拿货
    // pose 在指定区域，且手部姿态持续一段时间（或者gnc result）
    for (size_t i = 0; i < pose_results.size(); i++) {
        auto pose = pose_results[i]["pose"];
        // 手高于肩、持续时间
        if (pose[9 * 3 + 1] < pose[5 * 3 + 1] && pose[10 * 3 + 1] < pose[6 * 3 + 1]) {
            std::string event = "possible_pickup";
        }
        // debug
        // cout << camera_id << " = " << pose_results[i]["id"] << "\n";
        INFO("%s id = %s", camera_id.dump().c_str(), pose_results[i]["id"].dump().c_str());
        // auto box = pose_results[i]["box"];
        // cv::rectangle(image, cv::Point(box[0], box[1]), cv::Point(box[2], box[3]), cv::Scalar(0, 0, 255), 1);
    }
    // debug
    // if (pose_results.size() > 0) {
    //     cv::putText(image, to_string(frame_index), cv::Point(200, 100), 0, 1, cv::Scalar::all(0), 2, 16);
    //     cv::imwrite(cv::format("imgs_callback/%03d.jpg", (int)frame_index), image);
    // }
    // std::cout << "results is :" << pose_results << "\n";
}
void test_pipeline() {
    // debug
    // iLogger::rmtree("imgs");
    // iLogger::mkdir("imgs");
    // iLogger::rmtree("imgs_callback");
    // iLogger::mkdir("imgs_callback");
    std::string det_name  = "yolov5x-aqm";
    std::string pose_name = "yolov5s_pose";
    std::string gcn_name  = "";
    // std::vector<std::string> uris{"exp/39.mp4", "exp/37.mp4", "exp/38.mp4",
    //                               "exp/37.mp4", "exp/38.mp4", "rtsp://192.168.170.109:554/live/streamperson"};
    std::vector<std::string> uris{"rtsp://admin:admin123@192.168.170.109:580/cam/realmonitor?channel=4&subtype=0",
                                  "rtsp://admin:admin123@192.168.170.109:580/cam/realmonitor?channel=6&subtype=0"};

    auto pipeline = Pipeline::create_pipeline(det_name, pose_name, gcn_name);

    if (pipeline == nullptr) {
        std::cout << "pipeline create failed" << std::endl;
        return;
    }
    //
    pipeline->set_callback(callback);
    // auto out = pipeline->make_views(uris);
    for (auto uri : uris) {
        pipeline->make_view(uri);
    }

    // for (auto x : out) {
    //     std::cout << x << std::endl;
    // }

    auto current_uris = pipeline->get_uris();
    for (auto u : current_uris) {
        std::cout << u << std::endl;
    }
    // test disconnet
    // pipeline->disconnect_view("rtsp://192.168.170.109:554/live/streamperson6");
    current_uris = pipeline->get_uris();
    std::cout << "after disconnect_view(rtsp://192.168.170.109:554/live/streamperson6): "
              << "\n";
    for (auto u : current_uris) {
        std::cout << u << std::endl;
    }
}

int app_pipeline() {
    test_pipeline();
    return 0;
}